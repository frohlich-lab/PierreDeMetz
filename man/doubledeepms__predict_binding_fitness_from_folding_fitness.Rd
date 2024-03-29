% Generated by roxygen2: do not edit by hand
% Please edit documentation in
%   R/doubledeepms__predict_binding_fitness_from_folding_fitness.R
\name{doubledeepms__predict_binding_fitness_from_folding_fitness}
\alias{doubledeepms__predict_binding_fitness_from_folding_fitness}
\title{doubledeepms__predict_binding_fitness_from_folding_fitness}
\usage{

  doubledeepms__predict_binding_fitness_from_folding_fitness(folding_fitness,
  folding_linear_kernel, folding_linear_bias, binding_linear_kernel,
  binding_linear_bias, b_dg_wt, RT = 0.001987 * (273 + 24))
}
\arguments{
\item{folding_fitness}{Folding fitness (required)}

\item{folding_linear_kernel}{Folding fitness kernel (required)}

\item{folding_linear_bias}{Folding fitness bias (required)}

\item{binding_linear_kernel}{Binding fitness kernel (required)}

\item{binding_linear_bias}{Binding fitness bias (required)}

\item{b_dg_wt}{Wild-type binding delta G (required)}

\item{RT}{constant (default:0.001987*(273+24))}
}
\value{
Predicted binding fitness
}
\description{
Predict binding fitness from folding fitness (for no binding affinity model).
}
