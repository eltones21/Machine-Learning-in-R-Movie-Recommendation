# Movie-Recommendation-in-R
Movie Recommendation in R

This project completes part of the final assessment in HarvardX's Professional Certificate in Data Science. It explores the MovieLens 10M data set, focusing on the construction of models used to predict unknown ratings.

Graded Files
movieLens.R
This is the main R script which constructs the data and calculates predictions and RMSE results for each model. The RMSE of the final model, trained on the edx data set, when tested against the validation data set is defined as rmse_final. This script also creates some basic plots for data exploration, illustrated in the final report.

movieLensEC.Rmd
This is the final report in .Rmd form. The files loaded are created in main_code.R. The script code.R takes quite a while to run, so all of the R objects required to run final_report.Rmd are included in the folder movieLensEC_files. This folder must be present in the working directory (relative paths in line with this repo are used). This report is also available through this RPubs link.

movieLensEC.pdf
The final report in PDF format generated thorugh Latex using Knit.
