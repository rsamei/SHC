# Stat-Hackathon 2022 — predicting vehicle accident counts

First place, out of the teams that entered. The hackathon was organised by the
Department of Economics of the University of Bergamo on 31 May and 1 June 2022. There
were five of us; we used Python, R and SAS. This repository holds the Python part, which
is the part I worked on.

## The task

Given a vehicle, predict the number of accidents it causes. The data was 26,448
observations over 68 variables describing the vehicle itself — engine, dimensions,
weight, gearbox, body type, list price — with an exposure column and the accident count
as the response.

## What the code does

Most of the 68 variables are either numerical measurements or high-cardinality
categories, and a lot of them say much the same thing twice. So the first step is
reducing them.

`0_Multiple_correspondence_analysis.py` runs a greedy forward selection over the
categorical columns: at each round it tries every column not yet chosen, fits an MCA on
the set, and keeps whichever column adds the most explained inertia. It writes out the
ordering and the variance each step reached. `1_PCA_MCA.py` then takes the categorical
columns that cleared 0.25, fits PCA on the numerical block and MCA on that categorical
block, and produces the transformed coordinates.

`2_model_selection.ipynb` fits regressions on those coordinates with PyCaret, comparing
the model families and tuning the best one, and ends with PCA and t-SNE projections
coloured by the response to see whether the accident counts separate at all.

Two small cleaning decisions are in there as well: brand websites are cut down to their
top-level domain, and the validity date to its year, so neither becomes a category with
one level per row.

`common.py` holds the column lists.

## Running it

The competition dataset (`VehiclesDBV2.csv`) is not mine to publish, so it is not here.
The notebook outputs are committed, so the model comparison tables and the plots are
readable without it.

Python, prince, PyCaret, scikit-learn, pandas.
