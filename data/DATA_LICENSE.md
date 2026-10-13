# Observation data: sources and terms

We distinguish the observation-data terms below from the software license.
We distribute the processed Djougou and Medicina hourly CSVs and provide the
Membach source link and preprocessing program.

## Djougou and Medicina — CC BY 4.0

IGETS documents its datasets under Creative Commons Attribution4.0
International. License: https://creativecommons.org/licenses/by/4.0/
IGETS documentation: https://isdc.gfz.de/igets-data-base/documentation
Official database report: https://doi.org/10.2312/GFZ.b103-16087

We use the EOST Level 2 products (code 32), g_nofil channel. We aggregate valid
minute observations to hourly values, require 60 unique on-grid valid minutes
per hour, retain the longest continuous valid hourly segment and export
columns date,value. We apply no additional interpolation or tide removal.
The distributed CSVs are our processed derivatives of those observations.
We acknowledge the station data producers and EOST's Level 2 processing.

Djougou source citation:
Boy, J.-P., Rosat, S., Hinderer, J., and Littel, F.(2017).
Superconducting Gravimeter Data from Djougou — Level 1. GFZ Data Services.
https://doi.org/10.5880/igets.dj.l1.001

Medicina source citation:
Wziontek, H., Wolf, P., Nowak, I., Richter, B., Rülke, A., Schwahn, W.,
Wilmes, H., and Zerbini, S.(2017). Superconducting Gravimeter Data from
Medicina — Level 1. GFZ Data Services.
https://doi.org/10.5880/igets.mc.l1.001

The station DOIs identify the source observations. Our input products are
EOST Level 2, described at:
https://isdc.gfz.de/fileadmin/isdc/docs/EOSTproducts_v2.pdf

Please cite the relevant station dataset and:
Boy, J.-P., Barriot, J.-P., Förste, C., Voigt, C., and Wziontek, H.(2020).
Achievements of the First 4 Years of the International Geodynamics and Earth
Tide Service(IGETS)2015–2019.
https://doi.org/10.1007/1345_2020_94

When sharing or adapting these processed data, retain attribution and the
CC BY 4.0 license link, indicate processing changes, and preserve any source
copyright/attribution notices. Attribution does not imply provider endorsement.

## Membach — ROB data policy

Provider: Royal Observatory of Belgium, GWR-C021 superconducting gravimeter.
Source: https://seismology.be/en/gravimetry/observations/online-database
Data policy: https://www.seismology.be/en/legal-notices/data-policy

ROB's default policy permits private, educational and scientific use, requires
source URL and project/researcher attribution on copies and publications,
and requires written consent for commercial use. Specific project terms can
also apply. We provide an external download route and preprocessing script;
the public package excludes the Membach observation CSV. Readers use the
source observations under ROB's policy. See dataset/membach/README.md.
