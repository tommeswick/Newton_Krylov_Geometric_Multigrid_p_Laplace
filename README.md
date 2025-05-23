# Newton_Krylov_Geometric_Multigrid_p_Laplace

In this program examples from TouWi17 (SISC)
https://epubs.siam.org/doi/abs/10.1137/16M1067792
can be computed with a Newton-Krylov (CG) solver and geometric multigrid preconditioning.
The program is based and tested on deal.II version 9.6.0. Building and running the program is a usual when working with deal.II tutorial steps: cmake (to build once the Makefile), afterwards make run, i.e., make debug run or make release run.

The base program uses p=2, which corresponds to the classical Poisson problem: Find u:\Omega\to R such that -\Delta u=f in \Omega; here on the unit square \Omega=(0,1)^2, and homogeneous Dirichlet boundary conditions, and right hand side f=-1 . Changing test cases is done towards the end after `// Defining test cases'.

More information in terms of a bullet list are:
 - 3 test cases; grep for "Defining test cases"
 - A geometric multigrid method
 - But also switched (via "linear_solver_type") to a LU method (UMFPACK)
 - The nonlinear problem is solved with Newton's method
   Two Newton methods are implemented:
    - a) a standard line search (see paper TouWi17, SISC, 2017)
    - b) a modified Newton method in which 
        certain terms are scaled
 - Evaluation of several norms. The most important is 
   the F-norm that is equivalent to the H1 norm for the case p=2
 - Graphical output is written in *.vtk that can be displaced by paraview
   or visit.
 - Simulations of TouWi17 using this Newton-Krylov GMG method can be
   found in Wi22_NumPDE
   https://repo.uni-hannover.de/items/33ce4843-9a8d-480f-ad38-2c0bd0f9cae5
   p. 260: Section 10.8
   p. 333ff : Section 13.15

Contact: in case of questions, do not hesitate to contact me via email: thomas.wick@ifam.uni-hannover.de
