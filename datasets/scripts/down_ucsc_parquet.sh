# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]

#!/bin/bash
COUNT=${1:-2}
for ((i=0;i<COUNT;i++)) do
	python prepare/down_ucscvlaa_parquet.py -i $i -r $((i+1)) -l -o UCSC_VLAA
done
