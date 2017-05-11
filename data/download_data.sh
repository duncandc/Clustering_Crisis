#!/bin/bash

read -p "Are you sure you want to download all data (~10 Gb)? " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    curl -O --max-time 86400 "ftp://ftp.astro.yale.edu/pub/dac29/Clustering_Crisis_Data/bolshoi_additional_halo_properties.hdf5"
fi