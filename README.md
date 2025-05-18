# Keck NIRSPEC Helium Narrow-Band Flat Field Generator

This repository contains Python code for generating flat field calibration frames for the Keck NIRSPEC instrument using the helium narrow-band filter. These flats are tailored for infrared spectroscopy of the He I 10830 Å line, commonly used in studies of stellar winds, exoplanet atmospheres, and young stars.

## Features
- Combines and normalizes flat frames
- Applies dark subtraction and flat normalization
- Includes quality control visualizations

## Requirements
- Python 3.8+
- astropy
- numpy
- matplotlib
- glob
- tqdm

## Installation 
`
git clone https://github.com/yourusername/nirspec-helium-flats.git
cd nirspec-helium-flats
pip install -r requirements.txt
`

## 
