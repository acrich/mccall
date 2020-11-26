import subprocess
from string import Template

from model import Model
from validate import run_all
from wage_distribution import gen_plot, get_stats


"""
This script generates the tex/validation.pdf document.
The validation doc has a bunch of plots that show how the parameters interact,
and this is useful during calibration.
"""


# this runs all the scripts in the validation dir, and they generate
# all the graphs, and place them under results/
print("generating all the plots (this will take a long time)...")
#run_all()

# getting real parameters from the Model class, as we'll soon
# set them in the tex file template.
m = Model()

print("fetching wage statistics...")
wage_stats = get_stats()

print("generating template dictionary...")
d = {
    'z': m.z,
    'beta': m.β,
    'T': m.T,
    'alpha': round(m.α, 3),
    'mu': m.μ,
    'sigma': m.σ,
    'ism': m.ism,
    'c_hat': m.c_hat,
    'r': m.r,
    'w_size': m.w_size,
    'w_grid': m.w_grid,
    'w_draws': m.w_draws,
    'a_size': m.a_size,
    'a_min': m.a_min,
    'a_max': m.a_max,
    'w_min': m.w_min,
    'a_grid': m.a_grid,
    'rho': m.ρ,
    'wage_min': round(wage_stats['minimum'], 2),
    'wage_max': round(wage_stats['maximum'],2 ),
    'wage_mean': round(wage_stats['average'], 2),
    'wage_median': round(wage_stats['median'], 2),
}

print("reading template...")
# substituting all placeholders with model parameters fetched above.
with open('tex/validation.tex.tmpl', 'r') as f:
    src = Template(f.read())
    result = src.safe_substitute(d)

print("saving tex file...")
with open('tex/validation.tex', 'w+') as f:
    f.write(result)

print("generating wage plot...")
# generating wage distribution plot.
gen_plot()

print("converting tex to PDF...")
# generating PDF from tex.
# if this fails, maybe you need to install xelatex, like so:
# sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-generic-recommended
bashCommand = "xelatex validation"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd='tex/')
output, error = process.communicate()
