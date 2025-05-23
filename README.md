# Newton_Krylov_Geometric_Multigrid_p_Laplace

In this program examples from TouWi17 (SISC)
https://epubs.siam.org/doi/abs/10.1137/16M1067792
can be computed with a Newton-Krylov (CG) solver and geometric multigrid preconditioning.
The program is based and tested on deal.II version 9.6.0.

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
