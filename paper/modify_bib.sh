#!/bin/bash

sed -i -e 's/journal = {arXiv}/journal = {ArXiv e-prints}/g' bib.bib
sed -i -e 's/pages = {arXiv:.*//g' bib.bib
sed -i -e 's/eprinttype.*//g' bib.bib
sed -i -e 's/eprintclass.*//g' bib.bib
