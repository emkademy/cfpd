#!/usr/bin/env bash

mkdir -p ./data/models/
wget --no-check-certificate -O "./data/models/trained_using_augmented_images.pth" "https://onedrive.live.com/download?cid=8B924BF09E9CAC39&resid=8B924BF09E9CAC39%2133476&authkey=APgTamngqsiZkOw"
wget --no-check-certificate -O "./data/models/trained_using_normalized_images.pth" "https://onedrive.live.com/download?cid=8B924BF09E9CAC39&resid=8B924BF09E9CAC39%2133477&authkey=ALbbGqXaVsgT6oE"
wget --no-check-certificate -O "./data/models/trained_using_original_images.pth" "https://onedrive.live.com/download?cid=8B924BF09E9CAC39&resid=8B924BF09E9CAC39%2133478&authkey=AGI0d9AC4TntEd4"

