"""Loader for the larval antennal-lobe wiring diagram of Berck et al. (2016).

This module is responsible for loading and processing the datasheets consisting of the
wiring of both antenna lobes. The datasheet is read as an adjacency matrix.

This implementation as well as the data provided for it is based upon the work of
Berck et al 2016.

Source: Berck, Khandelwal et al. "The wiring diagram of a glomerular olfactory
system", eLife 5:e14859 (2016), https://doi.org/10.7554/eLife.14859, CC BY 4.0.
Supplementary files 1 and 2 are the left and right antennal-lobe adjacency
matrices. Download them from https://elifesciences.org/articles/14859/figures
into data/raw/berck2016/.

"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np
import openpyxl

# Get the spreadsheet containing the adjacency matrix for mapping out the neurons.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "berck2016"
FILES = {"left": "elife-14859-supp1-v2.xlsx", "right": "elife-14859-supp2-v2.xlsx"}
TOTAL_ROW = "DENDRITIC ARBOR TOTAL" # The last row in the spreadsheet.

## Citation to the original source and the provider of the spreadsheet.
CITATION = ("Berck ME, Khandelwal A, et al. (2016) The wiring diagram of a glomerular "
            "olfactory system. eLife 5:e14859. https://doi.org/10.7554/eLife.14859 (CC BY 4.0)")

# Class the neurons in to different classes depending on how their name appear in the spreadsheet.
def neuron_class(name: str) -> str:
    if " ORN " in f" {name} ":
        return "ORN"
    if name.startswith("mPN"):
        return "mPN"
    if name.startswith("Descending"):
        return "descending"
    if re.search(r"\bPN\b", name):
        return "uPN"
    if name.startswith(("Broad", "Keystone", "Picky", "Choosy", "Ventral LN")):
        return "LN"
    if name.startswith("SEZ"):
        return "SEZ"
    if name.startswith(("CSD", "lAL-1", "tdc")):
        return "neuromodulatory"
    return "other"

## Classify neurotransmitters in to four distinct classes, if none of them matches label it as unknown.
def transmitter(name: str) -> str:
    cls = neuron_class(name)
    if cls in ("ORN", "uPN"):
        return "excitatory"
    if name.startswith(("Broad", "Choosy", "Keystone")):
        return "GABA"
    if name.startswith("Picky"):
        return "glutamate"
    if cls == "neuromodulatory":
        return "modulatory"
    return "unknown"

# Returns the effect a neuron has on the neurons it connects to, based on its neurotransmitters. 
def sign(name: str) -> int:
    """+1 excitatory, -1 inhibitory (GABA; glutamate per Liu & Wilson 2013), 0 unknown."""
    return {"excitatory": 1, "GABA": -1, "glutamate": -1}.get(transmitter(name), 0)


def side_of(name: str) -> str:
    if re.search(r"\bleft\b", name):
        return "left"
    if re.search(r"\bright\b", name):
        return "right"
    return "midline"


def base_name(name: str) -> str:
    """Merge compartments: 'Picky 0 left [axon]' -> 'Picky 0 left'."""
    return re.sub(r"\s*\[(dendrites|axon)\]$", "", name)


@dataclass
class Lobe:
    """One antennal-lobe table: synapse counts M[pre, post] and dendritic input totals."""
    side: str
    names: list[str]
    matrix: np.ndarray
    dendritic_totals: np.ndarray


@dataclass
class Circuit:
    """Neuron-level graph of one or both lobes."""
    names: list[str]
    matrix: np.ndarray                      # synapses, rows presynaptic
    classes: list[str] = field(default_factory=list)
    signs: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def index(self, name: str) -> int:
        return self.names.index(name)


def read_lobe(side: str, data_dir: Path = DATA_DIR) -> Lobe:
    """This function is responsible for collecting input from spreadsheet files containing the neuron mappings.
        outputs a single Lobe data structure instance containing all relevant information.
    """

    path = data_dir / FILES[side]
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Make sure that it is downloaded correctly")
    
    rows = list(openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    header = list(rows[0][1:])
    keep = [i for i, name in enumerate(header) if name]
    names = [str(header[i]).strip() for i in keep]
    body = {str(r[0]).strip(): r[1:] for r in rows[1:] if r[0]}
    missing = [n for n in names if n not in body]
    
    if missing:
        raise ValueError(f"{path.name}: rows missing for {missing}")
    
    matrix = np.array([[float(body[n][i] or 0) for i in keep] for n in names])
    totals = np.array([float(body[TOTAL_ROW][i] or 0) for i in keep])
    
    if (matrix < 0).any() or (matrix != np.round(matrix)).any():
        raise ValueError(f"{path.name}: synapse counts must be non-negative integers")
    
    return Lobe(side, names, matrix, totals)


def merge_compartments(names: list[str], matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    # Some neurons appear as several rows/columns, one per compartment
    # ("Picky 0 left [axon]", "Picky 0 left [dendrites]"). Strip the compartment
    # suffix to get one name per neuron; dict.fromkeys removes duplicates but keeps
    # the order in which the neurons first appear in the table.
    merged = list(dict.fromkeys(base_name(n) for n in names))
    # Index of each neuron in the merged list.
    pos = {n: i for i, n in enumerate(merged)}
    # P maps compartments to neurons: P[i, j] = 1 if compartment i belongs to neuron j.
    P = np.zeros((len(names), len(merged)))
    for i, n in enumerate(names):
        P[i, pos[base_name(n)]] = 1
    # P.T @ matrix @ P sums the synapse counts over all compartments of the
    # presynaptic neuron (rows) and of the postsynaptic neuron (columns), giving a
    # neuron-to-neuron matrix with the same total number of synapses.
    return merged, P.T @ matrix @ P


def load_circuit(sides=("left", "right"), data_dir: Path = DATA_DIR) -> Circuit:
    """Combine lobes; each synapse lives in exactly one lobe, so counts add."""
    # Read each side's table and merge its compartments into whole neurons:
    # one (neuron names, synapse matrix) pair per lobe.
    parts = []
    for side in sides:
        lobe = read_lobe(side, data_dir)
        parts.append(merge_compartments(lobe.names, lobe.matrix))
    # Get all neurons of both lobes, each listed once. Neurons that innervate both lobes
    # appear in both tables but only get a single entry.
    names = list(dict.fromkeys(n for part_names, _ in parts for n in part_names))
    # Index of each neuron in the combined matrix.
    pos = {n: i for i, n in enumerate(names)}
    total = np.zeros((len(names), len(names)))
    for part_names, M in parts:
        # Where this lobe's neurons sit in the combined matrix.
        idx = [pos[n] for n in part_names]
        # Add the lobe's synapse counts into the matching rows and columns.
        # np.ix_ selects the full block of (pre, post) pairs between those indices.
        total[np.ix_(idx, idx)] += M
    # Label every neuron with its class and its sign
    return Circuit(names, total, [neuron_class(n) for n in names], np.array([sign(n) for n in names]))
