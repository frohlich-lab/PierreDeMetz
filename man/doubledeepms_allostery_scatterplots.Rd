% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/doubledeepms_allostery_scatterplots.R
\name{doubledeepms_allostery_scatterplots}
\alias{doubledeepms_allostery_scatterplots}
\title{doubledeepms_allostery_scatterplots}
\usage{
doubledeepms_allostery_scatterplots(input_file, temperature = 30,
  fitness_list, mochi_outpath_list, outpath, colour_scheme,
  execute = TRUE)
}
\arguments{
\item{input_file}{path to input file (required)}

\item{temperature}{temperature in degrees celcuis (default:30)}

\item{fitness_list}{list of folder paths to fitness data (required)}

\item{mochi_outpath_list}{list of paths to MoCHI thermo model fit results (required)}

\item{outpath}{output path for plots and saved objects (required)}

\item{colour_scheme}{colour scheme file (required)}

\item{execute}{whether or not to execute the analysis (default: TRUE)}
}
\value{
Nothing
}
\description{
Plot free energy scatterplots for allosteric mutations
}
