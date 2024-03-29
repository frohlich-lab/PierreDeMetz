% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/doubledeepms_protein_stability_plots.R
\name{doubledeepms_protein_stability_plots}
\alias{doubledeepms_protein_stability_plots}
\title{doubledeepms_protein_stability_plots}
\usage{
doubledeepms_protein_stability_plots(input_list, pdb_file_list,
  pdb_chain_query_list, aaprop_file, aaprop_file_selected, input_MSA_list,
  outpath, colour_scheme, execute = TRUE)
}
\arguments{
\item{input_list}{path to MoCHI thermo model fit results (required)}

\item{pdb_file_list}{path to PDB file (required)}

\item{pdb_chain_query_list}{query chain id (required)}

\item{aaprop_file}{path to amino acid properties file (required)}

\item{aaprop_file_selected}{path to file with selected subset of identifiers}

\item{input_MSA_list}{path to MSA frequencies data (required)}

\item{outpath}{output path for plots and saved objects (required)}

\item{colour_scheme}{colour scheme file (required)}

\item{execute}{whether or not to execute the analysis (default: TRUE)}
}
\value{
Nothing
}
\description{
Plot free energy heatmaps.
}
