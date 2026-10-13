from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')

# ============================================================
# Spectral-loss helpers
# ============================================================
def _rfft_float32(x):
    """
    Compute a numerically stable orthonormal RFFT along the forecast axis.

    Args:
        x: [B, pred_len, C]
    """
    if x.ndim != 3:
        raise ValueError(f"Expected [B, pred_len, C], got {tuple(x.shape)}")

    return torch.fft.rfft(
        x.float(),
        dim=1,
        norm="ortho",
    )


def complex_spectral_loss(
    prediction,
    target,
    exclude_dc=True,
):
    """
    Complex spectral distance.

    It jointly constrains spectral magnitude and phase without explicitly
    separating them into two objectives.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )

    pred_fft = _rfft_float32(prediction)
    target_fft = _rfft_float32(target)

    if exclude_dc and pred_fft.size(1) > 1:
        pred_fft = pred_fft[:, 1:, :]
        target_fft = target_fft[:, 1:, :]

    return torch.mean(torch.abs(pred_fft - target_fft))


def decomposed_spectral_loss(
    prediction,
    target,
    exclude_dc=True,
    eps=1e-8,
):
    """
    Return:
        magnitude_loss: log-magnitude Smooth-L1 loss
        phase_loss: magnitude-gated circular phase loss

    DC is excluded by default because MSE already constrains the sequence
    offset. For an even prediction length, the Nyquist bin is also excluded
    from the phase term because it is purely real.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )

    pred_fft = _rfft_float32(prediction)
    target_fft = _rfft_float32(target)

    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    mag_start = 1 if exclude_dc and pred_mag.size(1) > 1 else 0
    pred_mag_for_loss = pred_mag[:, mag_start:, :]
    target_mag_for_loss = target_mag[:, mag_start:, :]

    magnitude_loss = torch.nn.functional.smooth_l1_loss(
        torch.log1p(pred_mag_for_loss),
        torch.log1p(target_mag_for_loss),
    )

    # Phase always excludes DC. Exclude Nyquist for an even pred_len.
    phase_start = 1
    phase_end = -1 if (
        prediction.size(1) % 2 == 0 and pred_fft.size(1) > 2
    ) else None

    pred_phase_fft = pred_fft[:, phase_start:phase_end, :]
    target_phase_fft = target_fft[:, phase_start:phase_end, :]

    if pred_phase_fft.numel() == 0:
        phase_loss = prediction.float().sum() * 0.0
        return magnitude_loss, phase_loss

    pred_phase = torch.angle(pred_phase_fft)
    target_phase = torch.angle(target_phase_fft)
    phase_difference = pred_phase - target_phase

    target_phase_mag = torch.abs(target_phase_fft)
    reference = target_phase_mag.mean(
        dim=1,
        keepdim=True,
    ).clamp_min(eps)

    magnitude_ratio = target_phase_mag / reference
    phase_gate = magnitude_ratio / (magnitude_ratio + 1.0)

    circular_distance = 1.0 - torch.cos(phase_difference)
    phase_loss = (
        phase_gate * circular_distance
    ).sum() / phase_gate.sum().clamp_min(eps)

    return magnitude_loss, phase_loss



class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def _spectral_warmup_factor(self, epoch):
        """
        Linearly increase the spectral term during the first N epochs.

        spectral_warmup_epochs=0 disables warmup.
        """
        warmup_epochs = int(
            getattr(self.args, 'spectral_warmup_epochs', 0)
        )
        if warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / float(warmup_epochs))

    def _compose_training_loss(
        self,
        mse_loss,
        outputs,
        target,
        epoch,
    ):
        """
        Build the final training objective and return detached log values.

        Supported spectral_loss_type:
            complex
            decomposed
            magnitude
            phase
        """
        components = {
            'mse': float(mse_loss.detach().item()),
            'spectral': 0.0,
            'magnitude': 0.0,
            'phase': 0.0,
            'weighted_spectral': 0.0,
        }

        if not getattr(self.args, 'use_spectral_loss', False):
            return mse_loss, components

        if outputs.shape != target.shape:
            raise ValueError(
                f"Spectral loss requires equal shapes, got "
                f"{tuple(outputs.shape)} and {tuple(target.shape)}"
            )

        loss_type = str(
            getattr(self.args, 'spectral_loss_type', 'complex')
        ).lower()

        exclude_dc = not bool(
            getattr(self.args, 'spectral_include_dc', False)
        )
        warmup_factor = self._spectral_warmup_factor(epoch)

        # Keep FFT loss in float32 even when model forward uses AMP.
        with torch.cuda.amp.autocast(enabled=False):
            if loss_type == 'complex':
                spectral = complex_spectral_loss(
                    outputs,
                    target,
                    exclude_dc=exclude_dc,
                )
                lambda_spectral = float(
                    getattr(self.args, 'lambda_spectral', 0.05)
                )
                weighted = (
                    warmup_factor
                    * lambda_spectral
                    * spectral
                )
                total_loss = mse_loss + weighted

                components['spectral'] = float(
                    spectral.detach().item()
                )
                components['weighted_spectral'] = float(
                    weighted.detach().item()
                )

            elif loss_type in (
                'decomposed',
                'magnitude',
                'phase',
            ):
                magnitude, phase = decomposed_spectral_loss(
                    outputs,
                    target,
                    exclude_dc=exclude_dc,
                )

                lambda_mag = float(
                    getattr(self.args, 'lambda_mag', 0.05)
                )
                lambda_phase = float(
                    getattr(self.args, 'lambda_phase', 0.01)
                )

                if loss_type == 'decomposed':
                    weighted = warmup_factor * (
                        lambda_mag * magnitude
                        + lambda_phase * phase
                    )
                elif loss_type == 'magnitude':
                    weighted = (
                        warmup_factor
                        * lambda_mag
                        * magnitude
                    )
                else:
                    weighted = (
                        warmup_factor
                        * lambda_phase
                        * phase
                    )

                total_loss = mse_loss + weighted

                components['magnitude'] = float(
                    magnitude.detach().item()
                )
                components['phase'] = float(
                    phase.detach().item()
                )
                components['weighted_spectral'] = float(
                    weighted.detach().item()
                )

            else:
                raise ValueError(
                    "Unknown spectral_loss_type="
                    f"'{loss_type}'. Expected one of: "
                    "complex, decomposed, magnitude, phase."
                )

        return total_loss, components

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_mse_loss = []
            train_spectral_loss = []
            train_magnitude_loss = []
            train_phase_loss = []
            train_weighted_spectral = []

            self.model.train()
            epoch_time = time.time()
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]
                ).float()
                dec_inp = torch.cat(
                    [
                        batch_y[:, :self.args.label_len, :],
                        dec_inp,
                    ],
                    dim=1,
                ).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                        )
                        f_dim = (
                            -1
                            if self.args.features == 'MS'
                            else 0
                        )
                        outputs = outputs[
                            :,
                            -self.args.pred_len:,
                            f_dim:,
                        ]
                        target = batch_y[
                            :,
                            -self.args.pred_len:,
                            f_dim:,
                        ]
                        mse_loss = criterion(outputs, target)
                else:
                    outputs = self.model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                    )
                    f_dim = (
                        -1
                        if self.args.features == 'MS'
                        else 0
                    )
                    outputs = outputs[
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]
                    target = batch_y[
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]
                    mse_loss = criterion(outputs, target)

                loss, loss_parts = self._compose_training_loss(
                    mse_loss=mse_loss,
                    outputs=outputs,
                    target=target,
                    epoch=epoch,
                )

                train_loss.append(loss.item())
                train_mse_loss.append(loss_parts['mse'])
                train_spectral_loss.append(
                    loss_parts['spectral']
                )
                train_magnitude_loss.append(
                    loss_parts['magnitude']
                )
                train_phase_loss.append(loss_parts['phase'])
                train_weighted_spectral.append(
                    loss_parts['weighted_spectral']
                )

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | "
                        "total: {2:.7f} mse: {3:.7f} "
                        "weighted_spec: {4:.7f}".format(
                            i + 1,
                            epoch + 1,
                            loss.item(),
                            loss_parts['mse'],
                            loss_parts['weighted_spectral'],
                        )
                    )
                    speed = (
                        time.time() - time_now
                    ) / iter_count
                    left_time = speed * (
                        (
                            self.args.train_epochs - epoch
                        )
                        * train_steps
                        - i
                    )
                    print(
                        '\tspeed: {:.4f}s/iter; '
                        'left time: {:.4f}s'.format(
                            speed,
                            left_time,
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(
                "Epoch: {} cost time: {}".format(
                    epoch + 1,
                    time.time() - epoch_time,
                )
            )

            train_loss = np.average(train_loss)
            train_mse = np.average(train_mse_loss)
            train_spec = np.average(train_spectral_loss)
            train_mag = np.average(train_magnitude_loss)
            train_phase = np.average(train_phase_loss)
            train_weighted_spec = np.average(
                train_weighted_spectral
            )

            # Keep checkpoint selection based on validation MSE.
            vali_loss = self.vali(
                vali_data,
                vali_loader,
                criterion,
            )
            test_loss = self.vali(
                test_data,
                test_loader,
                criterion,
            )

            print(
                "Epoch: {0}, Steps: {1} | "
                "Train Total: {2:.7f} "
                "Train MSE: {3:.7f} "
                "Raw Complex: {4:.7f} "
                "Mag: {5:.7f} "
                "Phase: {6:.7f} "
                "Weighted Spec: {7:.7f} "
                "Vali MSE: {8:.7f} "
                "Test MSE: {9:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    train_mse,
                    train_spec,
                    train_mag,
                    train_phase,
                    train_weighted_spec,
                    vali_loss,
                    test_loss,
                )
            )

            early_stopping(
                vali_loss,
                self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(
                model_optim,
                epoch + 1,
                self.args,
            )
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)


        return
