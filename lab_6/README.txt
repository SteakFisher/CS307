WEEK 6 LAB: HOPFIELD NETWORKS
==============================

Authors: Abhijith Viju, Jayadeep Bejoy, Ziyan Solkar

FILES
-----
Core Implementation:
  - hopfield_network.py     Core Hopfield network classes
  - lab6_main.py            Main lab assignment (all 5 problems)
  - demo_simple.py          Demonstrations
  - requirements.txt        Python dependencies

Reports:
  - REPORT.md               Full report (Markdown)
  - report.tex              Full report (LaTeX/IEEE format)
  - ANSWERS.txt             Quick reference answers
  - compile_report.txt      Instructions for compiling reports

Output (generated):
  - problem1_associative_memory.png
  - problem2_capacity.png
  - problem3_error_correction.png
  - problem4_eight_rook.png
  - problem5_tsp.png
  - demo_letters.png
  - demo_energy.png

INSTALLATION
------------
pip install -r requirements.txt

USAGE
-----
Run complete lab:
  python lab6_main.py

Run demonstrations:
  python demo_simple.py

QUICK ANSWERS
-------------
Problem 3: Error Correction
  Answer: 15-20% noise reliably corrected

Problem 4: Eight-Rook Weights
  A = 2.0 (row constraint)
  B = 2.0 (column constraint)
  C = 1.0 (total count)
  Reason: A=B for symmetry, C lower as redundant

Problem 5: TSP Weights
  Answer: 10,000 weights
  Calculation: N^2 x N^2 = 100 x 100 = 10,000

RESULTS SUMMARY
---------------
Network: 100 neurons
Patterns stored: 5
Capacity (theoretical): ~14 patterns
Capacity (observed): ~23 patterns
Error correction: 15-20% noise
Eight-Rook weights: 4,096 (64x64)
TSP weights: 10,000 (100x100)

For detailed explanations, see REPORT.md or compile report.tex
