% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/doubledeepms_free_energy_heatmaps.R
\name{doubledeepms_free_energy_heatmaps}
\alias{doubledeepms_free_energy_heatmaps}
\title{doubledeepms_free_energy_heatmaps}
\usage{
doubledeepms_free_energy_heatmaps(input_file, domain_name, outpath,
  colour_scheme, plot_width = 10, plot_height = 4, execute = TRUE)
}
\arguments{
\item{input_file}{path to MoCHI thermo model fit results (required)}

\item{domain_name}{domain name (required)}

\item{outpath}{output path for plots and saved objects (required)}

\item{colour_scheme}{colour scheme file (required)}

\item{plot_width}{heatmap plot width in inches (default:10)}

\item{plot_height}{heatmap plot height in inches (default:4)}

\item{execute}{whether or not to execute the analysis (default: TRUE)}
}
\value{
Nothing
}
\description{
Plot free energy heatmaps.
}
