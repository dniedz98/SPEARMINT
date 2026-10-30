\# SPEARMINT



\*\*SPEARMINT\*\* constructs locally \*ab initio\* Wannier Hamiltonians for incommensurate materials using a nearsightedness-based simultaneous interpolation and extrapolation scheme as described in our preprint.



The method extracts Wannier matrix elements from density functional theory (DFT) calculations on finite MINT-proxy systems and then constructs Hamiltonians for arbitrarily large supplied geometries that represent potentially non-crystalline materials interfaces. Note that this method requires O(1) DFT for arbitrarily large system approximants.



\*\*Preprint:\*\* \[arXiv:2609.05082](https://arxiv.org/abs/2609.05082)



\## Repository Contents



\- \*\*`MINpuT.py`\*\* — Generates initial structures of MINT-proxy systems from corresponding input files. These structures must be relaxed and Wannierized using a DFT workflow not provided in this repository.

\- \*\*`ExtractWannier.py`\*\* — Extracts the relevant, most bulk-like, Wannier Hamiltonian matrix elements from Wannier code outputs.

\- \*\*`FormHamiltonian.py`\*\* — Constructs a locally \*ab initio\* Wannier Hamiltonian for a supplied large-system geometry. A local linear regression scheme simultaneously interpolates and extrapolates the extracted matrix elements to their bulk values across different local atomic environments.

\- \*\*`calcQDOS\_3D.py`\*\* — Calculates the 3D momentum-resolved density of states (qDOS) of the resulting Wannier Hamiltonian. This quantity remains well-defined for non-crystalline materials and is related to angle-resolved photoemission spectroscopy (ARPES) measurements, up to optical matrix elements.



Jupyter notebooks provide analogous workflows to the associated Python scripts.



\## Workflow



1\. \*\*Generate proxy structures:\*\* Use `MINpuT.py` to generate the input structures for DFT calculations that sample all local interlayer registries with sufficient density.

2\. \*\*Calculate Wannier matrix elements:\*\* Relax and Wannierize the proxy structures using your chosen DFT workflow.

3\. \*\*Extract matrix elements:\*\* Use `ExtractWannier.py` to collect the relevant central matrix elements.

4\. \*\*Construct the Hamiltonian:\*\* Use `FormHamiltonian.py` to build a locally \*ab initio\* Wannier Hamiltonian for the target geometry.

5\. \*\*Calculate spectral properties:\*\* Use `calcQDOS\_3D.py` to calculate the momentum-resolved density of states.



\## Requirements and Usage



Input formats, software dependencies, and detailed usage instructions will be documented as the repository develops. DFT calculations and Wannierization require external software. 



This is research code! For now:  

Only JDFTx(https://jdftx.org/) formats are supported for by "ExtractWannier.py". 

Parameters in the python files, such as file paths, must be edited manually by the user.



\## Citation



If you use SPEARMINT in your research, please cite the associated preprint or the published paper when available:



> \[SPEARMINT: arXiv:2609.05082](https://arxiv.org/abs/2609.05082)



\## License



This project is licensed under the \[MIT License](LICENSE).

