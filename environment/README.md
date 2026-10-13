# Environment

We conducted our experiments with Python 3.11 and torch 2.5.1+cu121 on an NVIDIA V100.
Install torch separately for the target hardware, then install requirements.txt.
pip_freeze_full.txt records our complete experiment environment; the root
requirements.txt lists the direct dependencies used by the forecasting path.
Bash wrappers require Linux, macOS, WSL or Git Bash. The Python launchers can
also be called directly on Windows. Our environment snapshot lists the software versions used for the paper.
