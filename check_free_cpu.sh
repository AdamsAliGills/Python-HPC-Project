#!/bin/bash

list_hpc_models() {
  join <(lshosts -l | awk '/HOST_NAME/ {h=$2} /MODEL/ {m=$2} /NCORES/ {if(h!="" && m!="") print h, m}' | sort) \
    <(bhosts | awk 'NR>1 && ($4-$5) > 0 {print $1}' | sort) |
    awk '{print $2}' | sort -u
}

list_hpc_models
