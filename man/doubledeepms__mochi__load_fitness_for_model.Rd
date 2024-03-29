% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/doubledeepms__mochi__load_fitness_for_model.R
\name{doubledeepms__mochi__load_fitness_for_model}
\alias{doubledeepms__mochi__load_fitness_for_model}
\title{doubledeepms__mochi__load_fitness_for_model}
\usage{
doubledeepms__mochi__load_fitness_for_model(dimsum_RData_file,
  order_subset = 0:2, sequence_type = "aminoacid",
  mean_input_count_threshold = 0)
}
\arguments{
\item{dimsum_RData_file}{DiMSum .RData file (required)}

\item{order_subset}{Subset of mutation orders (default:0:2)}

\item{sequence_type}{Sequence type: 'nucleotide' or 'aminoacid' (default:aminoacid)}

\item{mean_input_count_threshold}{Mean input count threshold (default:0)}
}
\value{
Data.table of fitness
}
\description{
Load fitness to be used for model fitting.
}
