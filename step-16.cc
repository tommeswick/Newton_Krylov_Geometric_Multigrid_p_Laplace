// Thomas Wick
// Leibniz University Hannover
// www.thomaswick.org
// May 23, 2025 (Version 3)
//
// Ecole Polytechnique
// Mar 31, 2017 (Version 1b)
// Apr 20, 2017 (Version 2)
//
//
// RICAM Linz
// Jul 5, 2016 (Version 1)

// README: Please read the following
// commented lines for a brief introduction

// Version 1: Until Apr 20, 2017
// This program is based on deal.II step-16 
// This program is based on the deal.II 8.1.0 version.


// Version 2: After Apr 20, 2017
// This program is based on deal.II step-16 / step-56
// This program is based on the deal.II 8.5.0 version.

// Version 3: After May 23, 2025
// This program is based on the deal.II 9.6.0 version.

// Where to put this file?
// Easiest way is to create in the deal.II/examples folder
// another folder, e.g., step-multigrid
// Therein you copy this file, the *.inp mesh files
// and the CMakeLists.txt

// Building a Makefile
// Type in the terminal: 
// cmake .

// Running the program in the terminal:
// make run // or make release run
// --> prints detailed output on the screen

// make run | grep Info
// --> prints all necessary output for `our' purposes
// --> Be careful, because one does not detect directly 
//     whether Newton or the linear solver diverge!!


// At some places you find TODO that explains further things


// Specific features of this program
// - Solves p-Laplace problem as discussed in TouWi17 SISC
//   https://epubs.siam.org/doi/abs/10.1137/16M1067792
//
// - 3 test cases; grep for "Defining test cases"
// - A geometric multigrid method
// - But also switched (via "linear_solver_type") to a LU method (UMFPACK)
// - The nonlinear problem is solved with Newton's method
//   Two Newton methods are implemented:
//     a) a standard line search (see paper TouWi17, SISC, 2017)
//     b) a modified Newton method in which 
//        certain terms are scaled
// - Evaluation of several norms. The most important is 
//   the F-norm that is equivalent to the H1 norm for the case p=2
// - Graphical output is written in *.vtk that can be displaced by paraview
//   or visit.
// - Simulations of TouWi17 using this Newton-Krylov GMG method can be
//   found in Wi22_NumPDE
//   https://repo.uni-hannover.de/items/33ce4843-9a8d-480f-ad38-2c0bd0f9cae5
//   p. 260: Section 10.8
//   p. 333ff : Section 13.15

// Here is the remaining introduction from the original step-16 program:
//
// As discussed in the introduction, most of this program is copied almost
// verbatim from step-6, which itself is only a slight modification of
// step-5. Consequently, a significant part of this program is not new if
// you've read all the material up to step-6, and we won't comment on that
// part of the functionality that is unchanged. Rather, we will focus on those
// aspects of the program that have to do with the multigrid functionality
// which forms the new aspect of this tutorial program.


// Again, the first few include files are already known, so we won't comment
// on them:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>  

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_iterator.h>
//#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>


// These, now, are the include necessary for the multilevel methods. The
// first two declare classes that allow us to enumerate degrees of freedom not
// only on the finest mesh level, but also on intermediate levels (that's what
// the MGDoFHandler class does) as well as allow to access this information
// (iterators and accessors over these cells).
//
// The rest of the include files deals with the mechanics of multigrid as a
// linear operator (solver or preconditioner).
//#include <deal.II/multigrid/mg_dof_handler.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>


// This is C++:
#include <fstream>
#include <sstream>


using namespace dealii;

template <int dim>
class DirichletBoundaryConditions : public Function<dim> 
{
  public:
  DirichletBoundaryConditions (const unsigned int test_case,
			       const double gamma,
			       const double omega)    
    : Function<dim>(1) 
    {
      _test_case = test_case;
      _gamma = gamma;
      _omega = omega;
    }
    
  virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;

  virtual void vector_value (const Point<dim> &p, 
			     Vector<double>   &value) const;

private:
  unsigned int _test_case;
  double _gamma, _omega;

};

// The boundary values are given to component 
// with number 0.
template <int dim>
double
DirichletBoundaryConditions<dim>::value (const Point<dim>  &p,
			     const unsigned int component) const
{
  Assert (component < this->n_components,
	  ExcIndexRange (component, 0, this->n_components));
  //double omega = 2.0 * M_PI;

  if (component == 0)
    {
      if (_test_case == 1)
	{
	  if (p(0) == 0 || 
	      p(0) == 3.14159265358979 || 
	      p(1) == 0 || 
	      p(1) == 3.14159265358979) 
	    return std::sin(p(0));
	  else 
	    return 0.0;
	}
      else if (_test_case == 2)
	{
	  if (p(0) == 0 || 
	      p(0) == 1 || 
	      p(1) == 0 || 
	      p(1) == 1 )
	    {
	      // Example 2 from the paper TouWi2017
	      return std::sin(_omega*(p(0) + p(1)));
	    }
	  else 
	    return 0.0;
	}
      else if (_test_case == 3)
	{
	  // Solution 3
	  if (p(0) == -0.5 || 
	      p(0) == 0.5 || //M_PI/2.0 || 
	      p(1) == -0.5 || 
	      p(1) == 0.5 )//M_PI/2.0) 
	    {
	      // Solution 3
	      return std::pow(p(0)*p(0) + p(1)*p(1), _gamma/2.0);
	    }
	  else
	    return 0.0;


	}

    }
  
  return 0.0;

}



template <int dim>
void
DirichletBoundaryConditions<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const 
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values (c) = DirichletBoundaryConditions<dim>::value (p, c);
}

/**************************************************************************/
template <int dim>
  class ManufacturedSolution : public Function<dim>
  {
    public:
    ManufacturedSolution (const unsigned int test_case,
			  const double gamma,
			  const double omega)
          :
		      Function<dim>(1)
      {
	_test_case = test_case;
	_gamma = gamma;
	_omega = omega;

      }

      virtual double
      value (
          const Point<dim> &p, const unsigned int component = 0) const
	{

	  
	  if (component == 0)
	    {
	      if (_test_case == 1)
		{
		  // Solution 1 (Example 1)
		  return std::sin(p(0));
		}
	      else if (_test_case == 2)
		{
		  // Solution 2 (Example 2)
		  return std::sin(_omega*(p(0) + p(1)));
		}
	      else if (_test_case == 3)
		{
		  // Solution 3
		  return std::pow(p(0)*p(0) + p(1)*p(1), _gamma/2.0);

		}
	    } // end component 0

	  return 0.0;
	}


    virtual Tensor<1,dim> 
    gradient (const Point<dim>   &p, const unsigned int  component = 0) const
    {
      Tensor<1,dim> return_value;
      return_value.clear();

      if (component == 0)
	{
	  if (_test_case == 1)
	    {
	      // Solution 1
	      return_value[0] = std::cos(p(0));
	      return_value[1] = 0.0;

	      return return_value;
	    }
	  else if (_test_case == 2)
	    {
	      // Solution 2
	      return_value[0] = _omega * std::cos(_omega * (p(0) + p(1)));
	      return_value[1] = _omega * std::cos(_omega * (p(0) + p(1)));
	      
	      return return_value;
	    }
	  else if (_test_case == 3)
	    {
	      // Solution 3
	      return_value[0] = _gamma * p(0) * std::pow(p(0)*p(0) + p(1)*p(1), (_gamma - 2.0)/2.0);
	      return_value[1] = _gamma * p(1) * std::pow(p(0)*p(0) + p(1)*p(1), (_gamma - 2.0)/2.0);
	      
	      return return_value;

	    }

	  
	} // end component 0
      
      return return_value;
      
    } // end gradient
    
  private:
    double _gamma;
    double _omega;
    unsigned int _test_case;
    
  };



// This main class is basically the same class as step-6, step-16, 
// and the nonlinear multiphysics template
//
//   Thomas Wick; ANS, Vol. 1, 2013
//   http://media.archnumsoft.org/10305/
//
//
// As far as
// member functions is concerned, the only addition is the
// <code>assemble_multigrid</code> function that assembles the matrices that
// correspond to the discrete operators on intermediate levels:
  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem (const unsigned int degree);
    void run ();

  private:
    // Setting up global runtime parameters
    void set_runtime_parameters_example_1 ();
    void set_runtime_parameters_example_2 ();
    void set_runtime_parameters_example_3 ();

    // Distribute DoFs, Newton bc, and MG initializations 
    void setup_system ();

    // Assemble the Jacobian (left hand side of Newton's method)
    void assemble_system_matrix ();   

    // Assemble residual (right hand side of Newton's method)
    void assemble_system_rhs ();
 
    // Assemble multigrid matrices, i.e., the 
    // linearized residual (similar to the Jacobian)
    void assemble_multigrid ();

    // Linear solver
    void solve ();
    
    // Nonlinear solver - developed in 
    // http://media.archnumsoft.org/10305/
    void newton_iteration();

    // Non-homogeneous Dirichlet conditions for the initial Newton solution
    void set_initial_bc ();

    // Not used here - initialized directly in setup_system ();
    void set_newton_bc ();
 
    // Heuristic mesh refinement - not used currently for the p-Laplace problem
    void refine_grid ();

    // Compute L2, F-norm (p-Laplace) errors etc.
    void compute_functional_values (); 

    // Implementation of the F-Norm
    void integrate_difference_F_norm 
    (const DoFHandler<dim> &dof,
     const Vector<double> &fe_function,
     const Function<dim> &exact_solution,
     Vector<float> &difference,
     const Quadrature<dim> &q,
     const Function<dim> *weight,
     const double alpha_eps,
     const double power_p_exponent);

    // Write vtk output
    void output_results (const unsigned int cycle) const;

    Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>    mg_dof_handler;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    // We need an additional object for the hanging nodes constraints. They
    // are handed to the transfer object in the multigrid. Since we call a
    // compress inside the multigrid these constraints are not allowed to be
    // inhomogeneous so we store them in different ConstraintMatrix objects.
    //ConstraintMatrix     hanging_node_constraints;
    //ConstraintMatrix     constraints;
    AffineConstraints<double> constraints;
    AffineConstraints<double> hanging_node_constraints;

    Vector<double>       solution, newton_update;
    Vector<double>       system_rhs;

    const unsigned int degree;

    // The following four objects are the only additional member variables,
    // compared to step-6. The first three represent the operators that act
    // on individual levels of the multilevel hierarchy, rather than on the
    // finest mesh as do the objects above while the last object stores
    // information about the boundary indices on each level and information
    // about indices lying on a refinement edge between two different
    // refinement levels.
    //
    // To facilitate having objects on each level of a multilevel hierarchy,
    // deal.II has the MGLevelObject class template that provides storage for
    // objects on each level. What we need here are matrices on each level,
    // which implies that we also need sparsity patterns on each level. As
    // outlined in the @ref mg_paper, the operators (matrices) that we need
    // are actually twofold: one on the interior of each level, and one at the
    // interface between each level and that part of the domain where the mesh
    // is coarser. In fact, we will need the latter in two versions: for the
    // direction from coarse to fine mesh and from fine to
    // coarse. Fortunately, however, we here have a self-adjoint problem for
    // which one of these is the transpose of the other, and so we only have
    // to build one; we choose the one from coarse to fine.
    MGLevelObject<SparsityPattern>       mg_sparsity_patterns;
    MGLevelObject<SparseMatrix<double> > mg_matrices;
    MGLevelObject<SparseMatrix<double> > mg_interface_matrices;
    MGConstrainedDoFs                    mg_constrained_dofs;


    // Some global variables - names are self-explaining
    std::string linear_solver_type;

    double cell_diameter, min_cell_diameter;
    double old_min_cell_diameter;
    double old_local_error_fnorm;


    unsigned max_no_refinement_cycles;
    unsigned int max_obtained_no_newton_steps;
    double lower_bound_newton_residual, lower_bound_linear_solver;

    int number_of_linear_iterations, min_number_of_linear_iterations, 
      max_number_of_linear_iterations;
    double omega, gamma;
    double power_of_norm, power_p;
    double force, alpha_eps, alpha_lambda;

    unsigned int  max_no_line_search_steps;

    bool bool_use_modified_Newton;
    double a_fp, b_fp, delta_fixed_point_newton;

    unsigned int test_case;

    double time_Newton_global_current_mesh;
  };



 


  // The constructor is left mostly unchanged. We take the polynomial degree
  // of the finite elements to be used as a constructor argument and store it
  // in a member variable.
  //
  // By convention, all adaptively refined triangulations in deal.II never
  // change by more than one level across a face between cells. For our
  // multigrid algorithms, however, we need a slightly stricter guarantee,
  // namely that the mesh also does not change by more than refinement level
  // across vertices that might connect two cells. In other words, we must
  // prevent the following situation:
  //
  //
  // This is achieved by passing the
  // Triangulation::limit_level_difference_at_vertices flag to the constructor
  // of the triangulation class.
  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem (const unsigned int degree)
    :
    triangulation (Triangulation<dim>::
                   limit_level_difference_at_vertices),
    // Q-functions: degree determines the polynomial order
    // Here: degree = 1, i.e., Q1
    fe (degree),
    mg_dof_handler (triangulation),
    degree(degree)
  {}




template <int dim>
void LaplaceProblem<dim>::set_runtime_parameters_example_1 ()
{
  // Lars Diening et al.:
  // p= 1.25, 4/3, 1.5, 5/3 1.8, 2, 2.25, 2.5, 3, 4
  power_p = 2; //1.01;
  power_of_norm = power_p - 2.0;

  omega = 2.0 * M_PI;
  gamma = 1.0;


  // Adrin Hirn model (squared norm and squared eps)
  // 1.1 <= p <= 3 => 1.0e-5
  // p <= 4 => 1.0e-4
  // p <= 5 => 1.0e-3
  alpha_eps = 1.0e-5;  //1.0e-4; //1.0e-5;

  // A Newton continuation method
  // But does not work very well. So,
  // keep this number as 0.0
  alpha_lambda = 0.0;

  std::string grid_name;
  //grid_name  = "unit_square_pi_half.inp"; 
  grid_name  = "unit_square_0_1.inp"; 

  
  GridIn<dim> grid_in;
  grid_in.attach_triangulation (triangulation);
  std::ifstream input_file(grid_name.c_str());      
  Assert (dim==2, ExcInternalError());
  grid_in.read_ucd (input_file); 
  
  triangulation.refine_global (1);

  max_no_refinement_cycles = 10; 


  // Tolerances for nonlinear and linear solver
  lower_bound_newton_residual = 1.0e-10; 
  lower_bound_linear_solver   = 1.0e-12;

  // Initializations (modified during computation)
  max_obtained_no_newton_steps = 0;
  old_min_cell_diameter = 1.0e-10;
  old_local_error_fnorm = 1.0e-10;
 
  number_of_linear_iterations     = 0;
  min_number_of_linear_iterations = 10000000; 
  max_number_of_linear_iterations = 0;

  // CG_with_MG_Prec, Direct
  linear_solver_type = "CG_with_MG_Prec";

  max_no_line_search_steps = 5;//100;
  bool_use_modified_Newton = false;
  a_fp = 0.01; 
  b_fp = 2.0; 

}



template <int dim>
void LaplaceProblem<dim>::set_runtime_parameters_example_2 ()
{
  // Diening et al.:
  // p= 1.25, 4/3, 1.5, 5/3 1.8, 2, 2.25, 2.5, 3, 4
  power_p = 1.01; //1.01;
  power_of_norm = power_p - 2.0;

  // 4 maxima: 2.0 * M_PI;
  omega = 1.0 * M_PI;
  gamma = 1.0;


  // Hirn model (squared norm and squared eps)
  // 1.1 <= p <= 3 => 1.0e-5
  // p <= 4 => 1.0e-4
  // p <= 5 => 1.0e-3
  alpha_eps = 1.0e-1;  //1.0e-4; //1.0e-5;

  // A Newton continuation method
  // But does not work very well. So,
  // keep this number as 0.0
  alpha_lambda = 0.0;

  std::string grid_name;
  grid_name  = "unit_square_0_1.inp"; 
  
  GridIn<dim> grid_in;
  grid_in.attach_triangulation (triangulation);
  std::ifstream input_file(grid_name.c_str());      
  Assert (dim==2, ExcInternalError());
  grid_in.read_ucd (input_file); 
  
  triangulation.refine_global (6);

  max_no_refinement_cycles = 6; 


  // Tolerances for nonlinear and linear solver
  lower_bound_newton_residual = 1.0e-10; // TODO: 1.0e-10
  lower_bound_linear_solver   = 1.0e-12;

  // Initializations (modified during computation)
  max_obtained_no_newton_steps = 0;
  old_min_cell_diameter = 1.0e-10;
  old_local_error_fnorm = 1.0e-10;
 
  number_of_linear_iterations     = 0;
  min_number_of_linear_iterations = 10000000; 
  max_number_of_linear_iterations = 0;

  // CG_with_MG_Prec, Direct
  linear_solver_type = "CG_with_MG_Prec";

  max_no_line_search_steps = 10;//100;
  bool_use_modified_Newton = false; // If true, set LS to 0
  a_fp = 0.01; 
  b_fp = 2.0; 

}


template <int dim>
void LaplaceProblem<dim>::set_runtime_parameters_example_3 ()
{
  // Diening et al.:
  // p= 1.25, 4/3, 1.5, 5/3 1.8, 2, 2.25, 2.5, 3, 4
  power_p = 1.5; //1.01;
  power_of_norm = power_p - 2.0;

  omega = 2.0 * M_PI;
  gamma = 0.8;


  // Hirn model (squared norm and squared eps)
  // 1.1 <= p <= 3 => 1.0e-5
  // p <= 4 => 1.0e-4
  // p <= 5 => 1.0e-3
  alpha_eps = 1.0;  //1.0e-4; //1.0e-5;

  // A Newton continuation method
  // But does not work very well. So,
  // keep this number as 0.0
  alpha_lambda = 0.0;

  std::string grid_name;
  //grid_name  = "unit_square.inp"; 
  grid_name  = "unit_square_05.inp"; 

  
  GridIn<dim> grid_in;
  grid_in.attach_triangulation (triangulation);
  std::ifstream input_file(grid_name.c_str());      
  Assert (dim==2, ExcInternalError());
  grid_in.read_ucd (input_file); 
  
  triangulation.refine_global (4);

  max_no_refinement_cycles = 6; 


  // Tolerances for nonlinear and linear solver
  lower_bound_newton_residual = 1.0e-10; 
  lower_bound_linear_solver   = 1.0e-12;

  // Initializations (modified during computation)
  max_obtained_no_newton_steps = 0;
  old_min_cell_diameter = 1.0e-10;
  old_local_error_fnorm = 1.0e-10;
 
  number_of_linear_iterations     = 0;
  min_number_of_linear_iterations = 10000000; 
  max_number_of_linear_iterations = 0;

  // CG_with_MG_Prec, Direct
  linear_solver_type = "CG_with_MG_Prec";

  max_no_line_search_steps = 0;//100;
  bool_use_modified_Newton = true;
  a_fp = 0.01; 
  b_fp = 2.0; 

}




  // The following function extends what the corresponding one in step-6
  // did. The top part, apart from the additional output, does the same:
  template <int dim>
  void LaplaceProblem<dim>::setup_system ()
  {
    mg_dof_handler.distribute_dofs (fe);
    mg_dof_handler.distribute_mg_dofs ();

    // Here we output not only the degrees of freedom on the finest level, but
    // also in the multilevel structure
    deallog << "Number of degrees of freedom: "
            << mg_dof_handler.n_dofs();

    for (unsigned int l=0; l<triangulation.n_levels(); ++l)
      deallog << "   " << 'L' << l << ": "
              << mg_dof_handler.n_dofs(l);
    deallog  << std::endl;

    sparsity_pattern.reinit (mg_dof_handler.n_dofs(),
                             mg_dof_handler.n_dofs(),
                             mg_dof_handler.max_couplings_between_dofs());
    DoFTools::make_sparsity_pattern (mg_dof_handler, sparsity_pattern);

    solution.reinit (mg_dof_handler.n_dofs());
    newton_update.reinit (mg_dof_handler.n_dofs());
    system_rhs.reinit (mg_dof_handler.n_dofs());

    // But it starts to be a wee bit different here, although this still
    // doesn't have anything to do with multigrid methods. step-6 took care of
    // boundary values and hanging nodes in a separate step after assembling
    // the global matrix from local contributions. This works, but the same
    // can be done in a slightly simpler way if we already take care of these
    // constraints at the time of copying local contributions into the global
    // matrix. To this end, we here do not just compute the constraints do to
    // hanging nodes, but also due to zero boundary conditions. We will use
    // this set of constraints later on to help us copy local contributions
    // correctly into the global linear system right away, without the need
    // for a later clean-up stage:
    constraints.clear ();
    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (mg_dof_handler, hanging_node_constraints);
    DoFTools::make_hanging_node_constraints (mg_dof_handler, constraints);


    /*
    // OLD
    // TODO: double-check whether homogeneous dirichlet bc
    // are appropriate for the multigrid assembly. I guess yes.
    std::set<types::boundary_id>         dirichlet_boundary_ids;
    typename FunctionMap<dim>::type      dirichlet_boundary;
    dealii::Functions::ZeroFunction<dim>                    homogeneous_dirichlet_bc (1);
    dirichlet_boundary_ids.insert(0);
    dirichlet_boundary[0] = &homogeneous_dirichlet_bc;


 
    // Set the Newton boundary conditions
    VectorTools::interpolate_boundary_values (static_cast<const DoFHandler<dim>&>(mg_dof_handler),
					      0,
					      dealii::Functions::ZeroFunction<dim>(1),  
					      constraints);
    */

    std::set<types::boundary_id> dirichlet_boundary_ids = {0};
    Functions::ZeroFunction<dim> homogeneous_dirichlet_bc;
    const std::map<types::boundary_id, const Function<dim> *>
      dirichlet_boundary_functions = {
        {types::boundary_id(0), &homogeneous_dirichlet_bc}};
    VectorTools::interpolate_boundary_values(mg_dof_handler,
                                             dirichlet_boundary_functions,
                                             constraints);

    constraints.close ();
    hanging_node_constraints.close ();
    constraints.condense (sparsity_pattern);
    sparsity_pattern.compress();
    system_matrix.reinit (sparsity_pattern);


    // Determine the minimal cell diameter for 
    // convergence order and other purposes
    min_cell_diameter = 1.0e+10;
    typename DoFHandler<dim>::active_cell_iterator
      cell = mg_dof_handler.begin_active(),
      endc = mg_dof_handler.end();
    
    for (; cell!=endc; ++cell)
      { 
	cell_diameter = cell->diameter();
	if (min_cell_diameter > cell_diameter)
	  min_cell_diameter = cell_diameter;	
	
      }



    // TODO: double-check bc for multigrid assembly:
    // Non-homogeneous or homogeneous Dirichlet ??
    // The multigrid constraints have to be initialized. They need to know
    // about the boundary values as well, so we pass the
    // <code>dirichlet_boundary</code> here as well.
    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(mg_dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(mg_dof_handler, dirichlet_boundary_ids);

    // Now for the things that concern the multigrid data structures. First,
    // we resize the multilevel objects to hold matrices and sparsity
    // patterns for every level. The coarse level is zero (this is mandatory
    // right now but may change in a future revision). Note that these
    // functions take a complete, inclusive range here (not a starting index
    // and size), so the finest level is <code>n_levels-1</code>.  We first
    // have to resize the container holding the SparseMatrix classes, since
    // they have to release their SparsityPattern before the can be destroyed
    // upon resizing.
    const unsigned int n_levels = triangulation.n_levels();

    mg_interface_matrices.resize(0, n_levels-1);
    mg_matrices.resize(0, n_levels-1);
    //mg_matrices.clear ();

    mg_sparsity_patterns.resize(0, n_levels-1);

    // Now, we have to provide a matrix on each level. To this end, we first
    // use the MGTools::make_sparsity_pattern function to first generate a
    // preliminary compressed sparsity pattern on each level (see the @ref
    // Sparsity module for more information on this topic) and then copy it
    // over to the one we really want. The next step is to initialize both
    // kinds of level matrices with these sparsity patterns.
    //
    // It may be worth pointing out that the interface matrices only have
    // entries for degrees of freedom that sit at or next to the interface
    // between coarser and finer levels of the mesh. They are therefore even
    // sparser than the matrices on the individual levels of our multigrid
    // hierarchy. If we were more concerned about memory usage (and possibly
    // the speed with which we can multiply with these matrices), we should
    // use separate and different sparsity patterns for these two kinds of
    // matrices.
    for (unsigned int level=0; level<n_levels; ++level)
      {
        DynamicSparsityPattern csp;
        csp.reinit(mg_dof_handler.n_dofs(level),
                   mg_dof_handler.n_dofs(level));
        MGTools::make_sparsity_pattern(mg_dof_handler, csp, level);

        mg_sparsity_patterns[level].copy_from (csp);

        mg_matrices[level].reinit(mg_sparsity_patterns[level]);
	mg_interface_matrices[level].reinit(mg_sparsity_patterns[level]);
      }
  }


// Assemble the left-hand side of Newton method, i.e., the Jacobian
  template <int dim>
  void LaplaceProblem<dim>::assemble_system_matrix ()
  {
    system_matrix=0;

    const QGauss<dim>  quadrature_formula(degree+1);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points  |  update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Vector<double> > old_solution_values (n_q_points, 
						      Vector<double>(1));
    
    std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, 
								  std::vector<Tensor<1,dim> > (1));


    // Declaring test functions:
    std::vector<double> phi_i_u (dofs_per_cell); 
    std::vector<Tensor<1,dim> > phi_i_grads_u(dofs_per_cell);


    typename DoFHandler<dim>::active_cell_iterator
    cell = mg_dof_handler.begin_active(),
    endc = mg_dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_matrix = 0;
        fe_values.reinit (cell);

	// Old Newton iteration values
	fe_values.get_function_values (solution, old_solution_values);
	fe_values.get_function_gradients (solution, old_solution_grads);
   

        for (unsigned int q=0; q<n_q_points; ++q)
	  {

	    for (unsigned int k=0; k<dofs_per_cell; ++k)
	      {
		phi_i_u[k]       = fe_values.shape_value (k, q);
		phi_i_grads_u[k] = fe_values.shape_grad (k, q);
	      }


	    Tensor<1,dim> grad_u = old_solution_grads[q][0];
	    
	    // Adaptive eps
//	    if ((grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1]) < 1.0e-4)
//	      alpha_eps = 1.0;
//	    else 
//	      alpha_eps = 1.0;

	    //	    double grad_u_norm = alpha_eps + std::sqrt(grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1]); 
	    
	    double grad_u_norm_squared = alpha_eps * alpha_eps + (grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1]);

	  
	  for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
		{

		  // Adrian Hirn power-law model
		  cell_matrix(j,i) += 
		    delta_fixed_point_newton * (power_of_norm/2.0 * std::pow(grad_u_norm_squared, (power_of_norm - 2)/2.0) *
		     2.0 * (grad_u * phi_i_grads_u[i]) * grad_u * phi_i_grads_u[j]
		     ) * fe_values.JxW(q);
		  
		  cell_matrix(j,i) +=  
		    (std::pow(grad_u_norm_squared, power_of_norm/2.0) *  
		     phi_i_grads_u[i]
		     ) * phi_i_grads_u[j] * fe_values.JxW(q);

		  // Newton continuation
		  cell_matrix(j,i) +=  alpha_lambda *
		    phi_i_grads_u[i] * phi_i_grads_u[j] * fe_values.JxW(q);

		}
         
            }

      }

        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global (cell_matrix,
                                                local_dof_indices,
                                                system_matrix);
      }
  }


// Assemble right-hand side of Newton's method, i.e., the residual
  template <int dim>
  void LaplaceProblem<dim>::assemble_system_rhs ()
  {
    system_rhs = 0;
    const QGauss<dim>  quadrature_formula(degree+1);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  
			     update_gradients |
                             update_quadrature_points  |  
			     update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Vector<double> > 
      old_solution_values (n_q_points, Vector<double>(1));
    
    std::vector<std::vector<Tensor<1,dim> > > 
      old_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (1));



    typename DoFHandler<dim>::active_cell_iterator
    cell = mg_dof_handler.begin_active(),
    endc = mg_dof_handler.end();
    for (; cell!=endc; ++cell)
      {
         cell_rhs = 0;
        fe_values.reinit (cell);

	fe_values.get_function_values (solution, old_solution_values);
	fe_values.get_function_gradients (solution, old_solution_grads);
    
	double grad_exact_u_norm_squared = 0.0;

        for (unsigned int q=0; q<n_q_points; ++q)
	  {

	    Tensor<1,dim> grad_u = old_solution_grads[q][0];

	    // Adaptive eps
//	    if ((grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1]) < 1.0e-4)
//	      alpha_eps = 1.0;
//	    else 
//	      alpha_eps = 1.0;
	    
	    // Check behavior of nonlinear function
	    //double tmp_tmp = std::pow(alpha_eps * alpha_eps + (grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1]), power_of_norm/2.0);
	    //std::cout << tmp_tmp << std::endl;

	    if (test_case == 1)
	      {
		grad_exact_u_norm_squared = 
		  alpha_eps * alpha_eps + std::cos(fe_values.quadrature_point(q)[0]) * std::cos(fe_values.quadrature_point(q)[0]);
		
		force  = std::pow(grad_exact_u_norm_squared, power_of_norm/2.0) * std::sin(fe_values.quadrature_point(q)[0])
		  + power_of_norm * std::pow(grad_exact_u_norm_squared, (power_of_norm - 2.0)/2.0) * 
		  std::cos(fe_values.quadrature_point(q)[0]) * std::cos(fe_values.quadrature_point(q)[0]) *
	      std::sin(fe_values.quadrature_point(q)[0]);
	      }
	    else if (test_case == 2)
	      {
		grad_exact_u_norm_squared = 
		  alpha_eps * alpha_eps + 2.0 * omega * omega * 
		  std::cos(omega * (fe_values.quadrature_point(q)[0] + fe_values.quadrature_point(q)[1])) * 
		  std::cos(omega * (fe_values.quadrature_point(q)[0] + fe_values.quadrature_point(q)[1])); 


		force = 2.0 * (4.0 * power_of_norm/2.0 * std::pow(grad_exact_u_norm_squared, (power_of_norm - 2.0)/2.0) * 
			 omega * omega * omega * omega * 
			 std::cos(omega * (fe_values.quadrature_point(q)[0] + fe_values.quadrature_point(q)[1])) * 
			 std::cos(omega * (fe_values.quadrature_point(q)[0] + fe_values.quadrature_point(q)[1])) * 
			 std::sin(omega * (fe_values.quadrature_point(q)[0] + fe_values.quadrature_point(q)[1]))
			 //
			 + omega * omega *
			 std::pow(grad_exact_u_norm_squared, power_of_norm/2.0) *
			 std::sin(omega * (fe_values.quadrature_point(q)[0] + fe_values.quadrature_point(q)[1]))
			 );

	      }
	      else if (test_case == 3)
	      {
	  // Solution 3
	  double aa = gamma * std::pow(fe_values.quadrature_point(q)[0] * fe_values.quadrature_point(q)[0] +
				       fe_values.quadrature_point(q)[1] * fe_values.quadrature_point(q)[1], (gamma - 2.0)/2.0);

	  force = 
	    // 1st term, 1st part
	    power_of_norm/2.0 * std::pow(alpha_eps * alpha_eps + 
				   (fe_values.quadrature_point(q)[0] * aa) * (fe_values.quadrature_point(q)[0] * aa) +
				   (fe_values.quadrature_point(q)[1] * aa) * (fe_values.quadrature_point(q)[1] * aa), (power_of_norm - 2.0)/2.0) * 
	    2.0 * fe_values.quadrature_point(q)[0] * gamma * gamma * (gamma - 1.0) * std::pow(fe_values.quadrature_point(q)[0] * fe_values.quadrature_point(q)[0] +
											       fe_values.quadrature_point(q)[1] * fe_values.quadrature_point(q)[1], gamma - 2.0) * 
	     // 1st term, 2nd part
	     (fe_values.quadrature_point(q)[0] * aa)
	     // 2nd term, 1st part
	     + std::pow(alpha_eps * alpha_eps + 
			(fe_values.quadrature_point(q)[0] * aa) * (fe_values.quadrature_point(q)[0] * aa) +
			(fe_values.quadrature_point(q)[1] * aa) * (fe_values.quadrature_point(q)[1] * aa), power_of_norm/2.0) * 
	     // 2nd term, 2nd part
	     (aa + 2.0 * fe_values.quadrature_point(q)[0] * fe_values.quadrature_point(q)[0] * gamma * (gamma - 2.0)/2.0 * 
	      std::pow(fe_values.quadrature_point(q)[0] * fe_values.quadrature_point(q)[0] +
		       fe_values.quadrature_point(q)[1] * fe_values.quadrature_point(q)[1], (gamma - 4.0)/2.0))
	     // 3rd term, 1st part
	     +  power_of_norm/2.0 * std::pow(alpha_eps * alpha_eps + 
					 (fe_values.quadrature_point(q)[0] * aa) * (fe_values.quadrature_point(q)[0] * aa) +
					 (fe_values.quadrature_point(q)[1] * aa) * (fe_values.quadrature_point(q)[1] * aa), (power_of_norm - 2.0)/2.0) * 
	     2.0 * fe_values.quadrature_point(q)[1] * gamma * gamma * (gamma - 1.0) * std::pow(fe_values.quadrature_point(q)[0] * fe_values.quadrature_point(q)[0] +
												fe_values.quadrature_point(q)[1] * fe_values.quadrature_point(q)[1], gamma - 2.0) * 
	      // 3rd term, 2nd part
	      (fe_values.quadrature_point(q)[1] * aa)
	       // 4th term, 1st part
	     + std::pow(alpha_eps * alpha_eps + 
			(fe_values.quadrature_point(q)[0] * aa) * (fe_values.quadrature_point(q)[0] * aa) +
			(fe_values.quadrature_point(q)[1] * aa) * (fe_values.quadrature_point(q)[1] * aa), power_of_norm/2.0) * 
	     // 2nd term, 2nd part
	     (aa + 2.0 * fe_values.quadrature_point(q)[1] * fe_values.quadrature_point(q)[1] * gamma * (gamma - 2.0)/2.0 * 
	      std::pow(fe_values.quadrature_point(q)[0] * fe_values.quadrature_point(q)[0] +
		       fe_values.quadrature_point(q)[1] * fe_values.quadrature_point(q)[1], (gamma - 4.0)/2.0));

	     



	    force *= -1.0;


	      }






	  //double grad_u_norm = alpha_eps + std::sqrt(grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1]); 
	  
	  double grad_u_norm_squared = alpha_eps * alpha_eps + (grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1]); 
	  
	  for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
	      const Tensor<1,dim> phi_i_grads_u = fe_values.shape_grad (i, q);
	      const double        phi_i_u = fe_values.shape_value (i, q);
	   

	      // Adrian Hirn power-law model
	      cell_rhs(i) -= (std::pow(grad_u_norm_squared, power_of_norm/2.0) * grad_u * phi_i_grads_u
			      //- force * phi_i_u
			      - 1.0 * phi_i_u
			       ) *  fe_values.JxW(q);

	      // Newton continuation
	      cell_rhs(i) -= (alpha_lambda * grad_u * phi_i_grads_u
			       ) *  fe_values.JxW(q);

            }

      }

        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global (cell_rhs,
                                                local_dof_indices,
                                                system_rhs);
      }
  }


  // The next function is the one that builds the linear operators (matrices)
  // that define the multigrid method on each level of the mesh. The
  // integration core is the same as above, but the loop below will go over
  // all existing cells instead of just the active ones, and the results must
  // be entered into the correct matrix. Note also that since we only do
  // multilevel preconditioning, no right-hand side needs to be assembled
  // here.
  //
  // Before we go there, however, we have to take care of a significant amount
  // of book keeping:
  template <int dim>
  void LaplaceProblem<dim>::assemble_multigrid ()
  {
    
    QGauss<dim>  quadrature_formula(1+degree);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | 
			     update_gradients |
                             update_quadrature_points | 
			     update_JxW_values);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Vector<double> > old_solution_values (n_q_points, 
						      Vector<double>(1));
    
    std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, 
								  std::vector<Tensor<1,dim> > (1));



    // Next a few things that are specific to building the multigrid data
    // structures (since we only need them in the current function, rather
    // than also elsewhere, we build them here instead of the
    // <code>setup_system</code> function). Some of the following may be a bit
    // obscure if you're not familiar with the algorithm actually implemented
    // in deal.II to support multilevel algorithms on adaptive meshes; if some
    // of the things below seem strange, take a look at the @ref mg_paper.
    //
    // Our first job is to identify those degrees of freedom on each level
    // that are located on interfaces between adaptively refined levels, and
    // those that lie on the interface but also on the exterior boundary of
    // the domain. As in many other parts of the library, we do this by using
    // Boolean masks, i.e. vectors of Booleans each element of which indicates
    // whether the corresponding degree of freedom index is an interface DoF
    // or not. The <code>MGConstraints</code> already computed the information
    // for us when we called initialize in <code>setup_system()</code>.

    // The indices just identified will later be used to decide where the
    // assembled value has to be added into on each level.  On the other hand,
    // we also have to impose zero boundary conditions on the external
    // boundary of each level. But this the <code>MGConstraints</code> knows.
    // So we simply ask for them by calling <code>get_boundary_indices()</code>.
    // The third step is to construct constraints on all those
    // degrees of freedom: their value should be zero after each application
    // of the level operators. To this end, we construct ConstraintMatrix
    // objects for each level, and add to each of these constraints for each
    // degree of freedom. Due to the way the ConstraintMatrix stores its data,
    // the function to add a constraint on a single degree of freedom and
    // force it to be zero is called Constraintmatrix::add_line(); doing so
    // for several degrees of freedom at once can be done using
    // Constraintmatrix::add_lines():
    std::vector<AffineConstraints<double>> boundary_constraints (triangulation.n_levels());
    std::vector<AffineConstraints<double>> boundary_interface_constraints (triangulation.n_levels());
    for (unsigned int level=0; level<triangulation.n_levels(); ++level)
      {
        boundary_constraints[level].add_lines (mg_constrained_dofs.get_refinement_edge_indices(level));
        boundary_constraints[level].add_lines (mg_constrained_dofs.get_boundary_indices(level));
        boundary_constraints[level].close ();

        IndexSet idx =
          mg_constrained_dofs.get_refinement_edge_indices(level)
          & mg_constrained_dofs.get_boundary_indices(level);

        boundary_interface_constraints[level].add_lines (idx);
        boundary_interface_constraints[level].close ();
      }

    
    // Declaring test functions:
    std::vector<double> phi_i_u (dofs_per_cell); 
    std::vector<Tensor<1,dim> > phi_i_grads_u(dofs_per_cell);


    // Now that we're done with most of our preliminaries, let's start the
    // integration loop. It looks mostly like the loop in
    // <code>assemble_system</code>, with two exceptions: (i) we don't need a
    // right hand side, and more significantly (ii) we don't just loop over
    // all active cells, but in fact all cells, active or not. Consequently,
    // the correct iterator to use is MGDoFHandler::cell_iterator rather than
    // MGDoFHandler::active_cell_iterator. Let's go about it:
    typename DoFHandler<dim>::cell_iterator cell = mg_dof_handler.begin(),
                                              endc = mg_dof_handler.end();

    for (; cell!=endc; ++cell)
      {
        cell_matrix = 0;
        fe_values.reinit (cell);

	// Old Newton iteration values
	fe_values.get_function_values (solution, old_solution_values);
	fe_values.get_function_gradients (solution, old_solution_grads);
  

        for (unsigned int q=0; q<n_q_points; ++q)
	  {
	    for (unsigned int k=0; k<dofs_per_cell; ++k)
	      {
		phi_i_u[k]       = fe_values.shape_value (k, q);
		phi_i_grads_u[k] = fe_values.shape_grad (k, q);
	      }
	    
	    Tensor<1,dim> grad_u = old_solution_grads[q][0];

	    // Adaptive eps
//	    if ((grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1]) < 1.0e-4)
//	      alpha_eps = 1.0;
//	    else 
//	      alpha_eps = 1.0;
	    

	    //	    double grad_u_norm = alpha_eps + std::sqrt(grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1]); 
	    
	    double grad_u_norm_squared = alpha_eps * alpha_eps + (grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1]);
	    

	    for (unsigned int i=0; i<dofs_per_cell; ++i)
	      {
		for (unsigned int j=0; j<dofs_per_cell; ++j)
		  {
		    // TODO (j,i) or (i,j) - double-check
		    // Maybe okay because the problem is symmetric
		    cell_matrix(j,i) += 
		      delta_fixed_point_newton * (power_of_norm/2.0 * std::pow(grad_u_norm_squared, (power_of_norm - 2)/2.0) *
		       2.0 * (grad_u * phi_i_grads_u[i]) * grad_u * phi_i_grads_u[j]
		       ) * fe_values.JxW(q);
		    
		    cell_matrix(j,i) += 
		      (std::pow(grad_u_norm_squared, power_of_norm/2.0) *  
		       phi_i_grads_u[i]
		       ) * phi_i_grads_u[j] * fe_values.JxW(q);
		  

		  }
	      }
	  }
	
        // The rest of the assembly is again slightly different. This starts
        // with a gotcha that is easily forgotten: The indices of global
        // degrees of freedom we want here are the ones for current level, not
        // for the global matrix. We therefore need the function
        // MGDoFAccessorLLget_mg_dof_indices, not
        // MGDoFAccessor::get_dof_indices as used in the assembly of the
        // global system:
        cell->get_mg_dof_indices (local_dof_indices);

        // Next, we need to copy local contributions into the level
        // objects. We can do this in the same way as in the global assembly,
        // using a constraint object that takes care of constrained degrees
        // (which here are only boundary nodes, as the individual levels have
        // no hanging node constraints). Note that the
        // <code>boundary_constraints</code> object makes sure that the level
        // matrices contains no contributions from degrees of freedom at the
        // interface between cells of different refinement level.
        boundary_constraints[cell->level()]
        .distribute_local_to_global (cell_matrix,
                                     local_dof_indices,
                                     mg_matrices[cell->level()]);

        // The next step is again slightly more obscure (but explained in the
        // @ref mg_paper): We need the remainder of the operator that we just
        // copied into the <code>mg_matrices</code> object, namely the part on
        // the interface between cells at the current level and cells one
        // level coarser. This matrix exists in two directions: for interior
        // DoFs (index $i$) of the current level to those sitting on the
        // interface (index $j$), and the other way around. Of course, since
        // we have a symmetric operator, one of these matrices is the
        // transpose of the other.
        //
        // The way we assemble these matrices is as follows: since the are
        // formed from parts of the local contributions, we first delete all
        // those parts of the local contributions that we are not interested
        // in, namely all those elements of the local matrix for which not $i$
        // is an interface DoF and $j$ is not. The result is one of the two
        // matrices that we are interested in, and we then copy it into the
        // <code>mg_interface_matrices</code> object. The
        // <code>boundary_interface_constraints</code> object at the same time
        // makes sure that we delete contributions from all degrees of freedom
        // that are not only on the interface but also on the external
        // boundary of the domain.
        //
        // The last part to remember is how to get the other matrix. Since it
        // is only the transpose, we will later (in the <code>solve()</code>
        // function) be able to just pass the transpose matrix where
        // necessary.
       for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            if (
              !mg_constrained_dofs.at_refinement_edge(cell->level(),
                                                      local_dof_indices[i])
              || mg_constrained_dofs.at_refinement_edge(cell->level(),
                                                        local_dof_indices[j])
            )
              cell_matrix(i,j) = 0;

        boundary_interface_constraints[cell->level()]
        .distribute_local_to_global (cell_matrix,
                                     local_dof_indices,
                                     mg_interface_matrices[cell->level()]);


      }
    
  }


// Impose non-homogeneous or homogeneous Dirichlet conditions
// The "initial" has nothing to do with 
// initial conditions for time-dependent problems, but the 
// boundary conditions for the first Newton iterate.
// After the first iteration, all non-homogeneous Dirichlet 
// conditions must be set to homogeneous conditions, which 
// is achieved in the function set_newton_bc()
template <int dim>
void
LaplaceProblem<dim>::set_initial_bc ()
{ 
  //std::map<unsigned int,double> boundary_values;
  std::map<types::global_dof_index,double> boundary_values;  
 
    // We apply non-homogeneous Dirichlet conditions
  VectorTools::interpolate_boundary_values (//mg_dof_handler,
					      static_cast<const DoFHandler<dim>&>(mg_dof_handler),
					      0,
					      dealii::Functions::ZeroFunction<dim>(1),  // Would be zero conditions
					      //DirichletBoundaryConditions<dim>(test_case,gamma,omega),
					      boundary_values);    

    /*
    for (typename std::map<unsigned int, double>::const_iterator
	   i = boundary_values.begin();
	 i != boundary_values.end();
	 ++i)
      solution(i->first) = i->second;
    */

    for (auto &boundary_value : boundary_values)
      solution(boundary_value.first) = boundary_value.second;
    
}



  // This is the other function that is significantly different in support of
  // the multigrid solver (or, in fact, the preconditioner for which we use
  // the multigrid method).
  //
  // Let us start out by setting up two of the components of multilevel
  // methods: transfer operators between levels, and a solver on the coarsest
  // level. In finite element methods, the transfer operators are derived from
  // the finite element function spaces involved and can often be computed in
  // a generic way independent of the problem under consideration. In that
  // case, we can use the MGTransferPrebuilt class that, given the constraints
  // on the global level and an MGDoFHandler object computes the matrices
  // corresponding to these transfer operators.
  //
  // The second part of the following lines deals with the coarse grid
  // solver. Since our coarse grid is very coarse indeed, we decide for a
  // direct solver (a Householder decomposition of the coarsest level matrix),
  // even if its implementation is not particularly sophisticated. If our
  // coarse mesh had many more cells than the five we have here, something
  // better suited would obviously be necessary here.
  template <int dim>
  void LaplaceProblem<dim>::solve ()
  {

    if (linear_solver_type == "CG_with_MG_Prec")
      {
    // Create the object that deals with the transfer between different
    // refinement levels. We need to pass it the hanging node constraints.

    MGTransferPrebuilt<Vector<double> > mg_transfer(mg_constrained_dofs);

    // This function is deprecated, but I did not understand, how 
    // hanging nodes in the future will be handled
    //    MGTransferPrebuilt<Vector<double> > mg_transfer(hanging_node_constraints, mg_constrained_dofs);


    // Now the prolongation matrix has to be built.  This matrix needs to take
    // the boundary values on each level into account and needs to know about
    // the indices at the refinement edges. The <code>MGConstraints</code>
    // knows about that so pass it as an argument.
    mg_transfer.build(mg_dof_handler);

    FullMatrix<double> coarse_matrix;
    coarse_matrix.copy_from (mg_matrices[0]);
    MGCoarseGridHouseholder<> coarse_grid_solver;
    coarse_grid_solver.initialize (coarse_matrix);

    // The next component of a multilevel solver or preconditioner is that we
    // need a smoother on each level. A common choice for this is to use the
    // application of a relaxation method (such as the SOR, Jacobi or
    // Richardson method) or a small number of iterations of a solver method
    // (such as CG or GMRES). The mg::SmootherRelaxation and
    // MGSmootherPrecondition classes provide support for these two kinds of
    // smoothers. Here, we opt for the application of a single SOR
    // iteration. To this end, we define an appropriate <code>typedef</code>
    // and then setup a smoother object.
    //
    // Since this smoother needs temporary vectors to store intermediate
    // results, we need to provide a VectorMemory object. Since these vectors
    // will be reused over and over, the GrowingVectorMemory is more time
    // efficient than the PrimitiveVectorMemory class in the current case.
    //
    // The last step is to initialize the smoother object with our level
    // matrices and to set some smoothing parameters.  The
    // <code>initialize()</code> function can optionally take additional
    // arguments that will be passed to the smoother object on each level. In
    // the current case for the SOR smoother, this could, for example, include
    // a relaxation parameter. However, we here leave these at their default
    // values. The call to <code>set_steps()</code> indicates that we will use
    // two pre- and two post-smoothing steps on each level; to use a variable
    // number of smoother steps on different levels, more options can be set
    // in the constructor call to the <code>mg_smoother</code> object.
    //
    // The last step results from the fact that we use the SOR method as a
    // smoother - which is not symmetric - but we use the conjugate gradient
    // iteration (which requires a symmetric preconditioner) below, we need to
    // let the multilevel preconditioner make sure that we get a symmetric
    // operator even for nonsymmetric smoothers:

    // TODO
    // - which smoother? Gauss-Seidel
    // - how many cycles?

    // Different smoother initializations
    // Relaxation parameter is 1 => Gauss-Seidel smoother
    typedef PreconditionSOR<SparseMatrix<double> > Smoother;
    //typedef PreconditionSSOR<SparseMatrix<double> > Smoother;
    //typedef PreconditionJacobi<SparseMatrix<double> > Smoother;
    
    mg::SmootherRelaxation<Smoother, Vector<double> > mg_smoother;
    mg_smoother.initialize(mg_matrices);
    mg_smoother.set_steps(2);
    mg_smoother.set_symmetric(true);

    // The next preparatory step is that we must wrap our level and interface
    // matrices in an object having the required multiplication functions. We
    // will create two objects for the interface objects going from coarse to
    // fine and the other way around; the multigrid algorithm will later use
    // the transpose operator for the latter operation, allowing us to
    // initialize both up and down versions of the operator with the matrices
    // we already built:
    mg::Matrix<Vector<double> > mg_matrix(mg_matrices);
    mg::Matrix<Vector<double> > mg_interface_up(mg_interface_matrices);
    mg::Matrix<Vector<double> > mg_interface_down(mg_interface_matrices);

    // Now, we are ready to set up the V-cycle operator and the multilevel
    // preconditioner.
    Multigrid<Vector<double> > mg(mg_matrix,
                                  coarse_grid_solver,
                                  mg_transfer,
                                  mg_smoother,
                                  mg_smoother);
    mg.set_edge_matrices(mg_interface_down, mg_interface_up);

    PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double> > >
    preconditioner(mg_dof_handler, mg, mg_transfer);

    // With all this together, we can finally get about solving the linear
    // system in the usual way. Here, 10000 is the maximum number 
    // iterations that we allow before the solver aborts.
    SolverControl solver_control (10000, lower_bound_linear_solver);

    // The problem is symmetric, thus CG is a good linear solver.
    // Otherwise just replace SolverCG<> by SolverGMRES<>
    SolverCG<>    cg (solver_control);

    //std::ofstream output ("text.txt");
    //system_matrix.print_formatted(output);
    //abort();

        cg.solve (system_matrix, newton_update, system_rhs,
                  preconditioner);

    // Sanity check with no preconditioner
    //    cg.solve (system_matrix, newton_update, system_rhs,
    //         PreconditionIdentity());

	// TODO: Somewhere we could then also
	// implement a pure multigrid solver

    number_of_linear_iterations = solver_control.last_step();


      }
    else if (linear_solver_type == "Direct")
      {
	Vector<double> sol, rhs;    
	sol = newton_update;    
	rhs = system_rhs;
	
	SparseDirectUMFPACK A_direct;
	A_direct.factorize(system_matrix);     
	A_direct.vmult(sol,rhs); 
	newton_update = sol;
	
	number_of_linear_iterations = 0;
      }

    constraints.distribute (newton_update);



  }


// The nonlinear solver. 
// This method has been more or less copied from
// http://media.archnumsoft.org/10305/
template <int dim>
void LaplaceProblem<dim>::newton_iteration () 
					       
{
  double time_for_Newton_current_mesh = 0.0;
  // NewIter = No of Newton iterations
  // LS = line search
  // Newt. = Newton
  // Res = Residual
  // Reduct = Reduction
  // BuiMat. = Build Jacobian matrix
  // FP = delta fixed point Newton when modified Newton method
  //      is used rather than a line search Newton method
  // Time = CPU per Newton step
  std::cout << "NewtIt.\t" << "Newt.Res.\t" << "Newt.Reduct\t"
	    << "BuiMat\t" << "LSIter\t" 
	    << "LinIter\t" << "FP\t\t" << "Time" << std::endl;


  Timer timer_newton;
  const unsigned int max_no_newton_steps  = 10;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 0.1; 
 
  // Line search parameters
  unsigned int line_search_step;
  //const unsigned int  max_no_line_search_steps = 0;//100;
  const double line_search_damping = 0.6;
  double new_newton_residual;
  
  // Application of the initial boundary conditions to the 
  // variational equations:
  set_initial_bc ();
  assemble_system_rhs();

  double newton_residual = system_rhs.linfty_norm(); 
  double old_newton_residual= newton_residual;
  double initial_newton_residual = newton_residual;
  unsigned int newton_step = 1;

  unsigned int stop_when_line_search_two_times_max_number = 0;
  
  max_obtained_no_newton_steps = newton_step;

  if (newton_residual < lower_bound_newton_residual)
    {
      std::cout << '\t' 
		<< std::scientific 
		<< newton_residual 
		<< std::endl;     
    }
  
  while ((newton_residual > lower_bound_newton_residual &&
	  (newton_residual/initial_newton_residual) > lower_bound_newton_residual) &&
	 newton_step < max_no_newton_steps)
    {
      timer_newton.start();
      old_newton_residual = newton_residual;
      
      assemble_system_rhs();
      newton_residual = system_rhs.linfty_norm();

      if (newton_residual < lower_bound_newton_residual)
	{
	  max_obtained_no_newton_steps = newton_step - 1;
	  std::cout << '\t' 
		    << std::scientific 
		    << newton_residual << std::endl;
	  break;
	}
  
      // Routine for intermediate quasi-Newton steps
      // but not used because I am not
      // sure how the multigrid assembly will be affected
      // when the matrix is not re-build
      //if (newton_residual/old_newton_residual > nonlinear_rho)
	assemble_system_matrix ();
	assemble_multigrid ();

      // Solve Ax = b
      solve ();	  
        
      // Backtracking line search
      line_search_step = 0;	  
      for ( ; 
	    line_search_step < max_no_line_search_steps; 
	    ++line_search_step)
	{	     					 
	  solution += newton_update;
	  
	  assemble_system_rhs ();			
	  new_newton_residual = system_rhs.linfty_norm();
	  
	  if (new_newton_residual < newton_residual)
	      break;
	  else 	  
	    solution -= newton_update;
	  
	  newton_update *= line_search_damping;
	}

      // Allow for increasing residual
      // TODO: be careful here. This works
      // for all examples in this program,
      // but there is no general proof of convergence
      //if (line_search_step == max_no_line_search_steps)
      //solution +=newton_update;

      if (line_search_step == max_no_line_search_steps)
	stop_when_line_search_two_times_max_number ++;
      
      if (stop_when_line_search_two_times_max_number == 3)
	{
	  std::cout << "Aborting Newton as line search does not help to converge anymore." << std::endl;
	  abort();
	}
	  
     
      timer_newton.stop();
      time_for_Newton_current_mesh += timer_newton.cpu_time ();
      
      std::cout << std::setprecision(5) <<newton_step << '\t' 
		<< std::scientific << newton_residual << '\t'
		<< std::scientific << newton_residual/old_newton_residual  <<'\t' ;
      if (newton_residual/old_newton_residual > nonlinear_rho)
	std::cout << "r" << '\t' ;
      else 
	std::cout << " " << '\t' ;
      std::cout << line_search_step  << '\t' 
		<< number_of_linear_iterations << '\t'
		<< delta_fixed_point_newton  << '\t'
		<< std::scientific << timer_newton.cpu_time () << '\t'
		<< std::scientific << time_for_Newton_current_mesh << '\t'
	//	<< alpha_lambda
		<< std::endl;

      

      // Compute max and min number of linear iterations per 
      // mesh level
      if (number_of_linear_iterations < min_number_of_linear_iterations)
	min_number_of_linear_iterations = number_of_linear_iterations;

      if (number_of_linear_iterations > max_number_of_linear_iterations)
	max_number_of_linear_iterations = number_of_linear_iterations;

      //alpha_lambda = alpha_lambda - 0.0001;

      if (bool_use_modified_Newton)
	{
	  // Update delta for dynamic switch between fixed point and Newton
	  double Qn = newton_residual/old_newton_residual;
	  double Qn_inv = old_newton_residual/newton_residual;
	  
	  delta_fixed_point_newton = delta_fixed_point_newton * (a_fp/(std::exp(Qn_inv)) + b_fp/(std::exp(Qn)));

	  // Normalize delta
	  if (delta_fixed_point_newton > 1.0)
	    delta_fixed_point_newton = 1.0;
	  else if (delta_fixed_point_newton < 0.0)
	    delta_fixed_point_newton = 0.0;
	  
	}

 

      // Updates
      timer_newton.reset();
      newton_step++;      
    }

  time_Newton_global_current_mesh = time_for_Newton_current_mesh;
  alpha_lambda = 0.0;

}



// Heuristic error estimator!!!
// Currently, not in use.
  template <int dim>
  void LaplaceProblem<dim>::refine_grid ()
  {
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate (static_cast<DoFHandler<dim>&>(mg_dof_handler),
                                        QGauss<dim-1>(3),
                                        //typename FunctionMap<dim>::type(),
					std::map<types::boundary_id, const Function<dim> *>(),
                                        solution,
                                        estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     0.3, 0.03);
    triangulation.execute_coarsening_and_refinement ();
  }



  template <int dim>
  void LaplaceProblem<dim>::output_results (const unsigned int cycle) const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler (mg_dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();

    std::ostringstream filename;
    filename << "solution-"
             << cycle
             << ".vtk";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
  }


// Compute error norms and convergence
template<int dim>
void LaplaceProblem<dim>::compute_functional_values()
{

  // L2 error
  ManufacturedSolution<dim> exact_u(test_case,gamma,omega);
  Vector<float> error_u (triangulation.n_active_cells());
  Vector<float> error_dxu (triangulation.n_active_cells());
  Vector<float> error_w1p (triangulation.n_active_cells());
  Vector<float> error_F (triangulation.n_active_cells());

  //  const ZeroFunction<dim>(1) zero;

  VectorTools::integrate_difference (mg_dof_handler,
				     solution,
				     exact_u,
				     error_u,
				     QGauss<dim>(fe.degree+1),
				     VectorTools::L2_norm
				     //&value_select
				     );
  double local_error_u = error_u.l2_norm();



 
  // H1 norm 
  VectorTools::integrate_difference (mg_dof_handler,
				     solution,
				     exact_u,
				     error_dxu,
				     QGauss<dim>(fe.degree+1),
				     VectorTools::H1_norm
				     //&value_select
				     );

  double local_error_uh1norm = error_dxu.l2_norm();

  // W1p norm 
  ComponentSelectFunction<dim> value_select (0, 1);
  VectorTools::integrate_difference (mg_dof_handler,
				     solution,
				     exact_u,
				     error_w1p,
				     QGauss<dim>(fe.degree+1),
				     VectorTools::W1p_norm,
				     &value_select,
				     power_p
				     );

  double local_error_w1pnorm = error_w1p.l2_norm();


  // F-norm (for p-Laplace)
  integrate_difference_F_norm (mg_dof_handler,
			       solution,
			       exact_u,
			       error_F,
			       QGauss<dim>(fe.degree+1),
			       &value_select,
			       alpha_eps, // alpha_eps
			       power_p); // power_law_exponent
  
  double local_error_fnorm = error_F.l2_norm();


 // Cut a part of the domain
  typename DoFHandler<dim>::active_cell_iterator
    cell = mg_dof_handler.begin_active(),
    endc = mg_dof_handler.end();
  
  unsigned int cell_counter = 0;
  double local_error_F = 0.0;

 for (; cell!=endc; ++cell)
   {    

     if (//(std::abs(cell->center()[1] - 0.0) >= 0.5 && cell->center()[0]  < 0.5) ||
	 cell->center()[0]  <= 1.25
	 )
       {
	 local_error_F  += error_F(cell_counter) * error_F(cell_counter);

       }

     cell_counter++;
   }

 // F norm u 
 local_error_F  = std::sqrt(local_error_F);

 if (test_case == 1)
   local_error_fnorm  = local_error_F;

  
  // Compute convergence rate
  // r = log(e_1/e_2) / log(h_1/h_2)

  double conv_rate = std::log(old_local_error_fnorm/local_error_fnorm) / std::log(old_min_cell_diameter/min_cell_diameter);
  
  std::cout << "Errors: L2 : " << local_error_u 
	    << "  H1 : " << local_error_uh1norm 
	    << "  W1p: " << local_error_w1pnorm 
	    << "  F: " << local_error_fnorm 
	    << "  CR: " << conv_rate
	    << "  h: "  << min_cell_diameter << std::endl;
  
  // CR = convergence rate for the F-norm

//  std::cout << "Info: " << min_cell_diameter << "  "  <<local_error_fnorm  << "  " << max_obtained_no_newton_steps << std::endl;

 std::cout << "Info:\t" 	<< std::scientific  
	    << triangulation.n_active_cells() << "\t" 
	    << mg_dof_handler.n_dofs() << "\t" 
	    << min_cell_diameter << "\t"  
   //<< local_error_fnorm  << "\t" 
   //<< conv_rate << "\t" 
	   << min_number_of_linear_iterations << ","
	   << max_number_of_linear_iterations << "\t\t"
	   << max_obtained_no_newton_steps << "\t\t"  
	   << time_Newton_global_current_mesh
	   << std::endl;

  // Reset min and max for linear iterations
  min_number_of_linear_iterations = 10000000; 
  max_number_of_linear_iterations = 0;

 

 old_min_cell_diameter = min_cell_diameter;
 old_local_error_fnorm = local_error_fnorm;


 // TODO: so wie hier, sollte dann wohl auch 
 // der Fehler/Konvergenz im Mehrgitter bestimmt werden koennen.

}


// Implementation of the F-Norm that is
// the natural measure for power-law type problems
template <int dim>
void LaplaceProblem<dim>::integrate_difference_F_norm 
(const DoFHandler<dim> &dof,
 const Vector<double> &fe_function,
 const Function<dim> &exact_solution,
 Vector<float> &difference,
 const Quadrature<dim> &q,
 const Function<dim> *weight,
 const double alpha_eps,
 const double power_p_exponent)
{

      double power_local = (power_p_exponent - 2.0)/4.0;
      double alpha_eps_squared_local = alpha_eps * alpha_eps;


      const unsigned int        n_components = dof.get_fe().n_components();
      const bool                fe_is_system = (n_components != 1);

      if (weight!=0)
        {
          Assert ((weight->n_components==1) || (weight->n_components==n_components),
                  ExcDimensionMismatch(weight->n_components, n_components));
        }

      difference.reinit (triangulation.n_active_cells());

      UpdateFlags update_flags = UpdateFlags (update_quadrature_points  |
                                              update_JxW_values);

          update_flags |= UpdateFlags (update_gradients);

 FEValues<dim> fe_values (fe, q, update_flags);

 const unsigned int max_n_q_points = q.size();

      std::vector< dealii::Vector<double> >
      function_values (max_n_q_points, dealii::Vector<double>(n_components));
      std::vector<std::vector<Tensor<1,dim> > >
      function_grads (max_n_q_points, std::vector<Tensor<1,dim> >(n_components));

      std::vector<double>
      weight_values (max_n_q_points);
      std::vector<dealii::Vector<double> >
      weight_vectors (max_n_q_points, dealii::Vector<double>(n_components));

      std::vector<dealii::Vector<double> >
      psi_values (max_n_q_points, dealii::Vector<double>(n_components));
      std::vector<std::vector<Tensor<1,dim> > >
      psi_grads (max_n_q_points, std::vector<Tensor<1,dim> >(n_components));

      std::vector<std::vector<Tensor<1,dim> > >
      psi_grads_tmp (max_n_q_points, std::vector<Tensor<1,dim> >(n_components));


      std::vector<double>
      psi_scalar (max_n_q_points);

      // tmp vector when we use the
      // Function<dim> functions for
      // scalar functions
      std::vector<double>         tmp_values (max_n_q_points);
      std::vector<Tensor<1,dim> > tmp_gradients (max_n_q_points);

      // loop over all cells
      typename DoFHandler<dim>::active_cell_iterator cell = dof.begin_active(),
                                        endc = dof.end();

      
      for (unsigned int index=0; cell != endc; ++cell, ++index)
        if (cell->is_locally_owned())
          {
            double diff=0;
            // initialize for this cell
            //x_fe_values.reinit (cell);
	    fe_values.reinit(cell);

           
            const unsigned int   n_q_points = fe_values.n_quadrature_points;

            // resize all out scratch
            // arrays to the number of
            // quadrature points we use
            // for the present cell
            function_values.resize (n_q_points,
                                    dealii::Vector<double>(n_components));
            function_grads.resize (n_q_points,
                                   std::vector<Tensor<1,dim> >(n_components));

            weight_values.resize (n_q_points);
            weight_vectors.resize (n_q_points,
                                   dealii::Vector<double>(n_components));

            psi_values.resize (n_q_points,
                               dealii::Vector<double>(n_components));
            psi_grads.resize (n_q_points,
                              std::vector<Tensor<1,dim> >(n_components));

	    psi_grads_tmp.resize (n_q_points,
                              std::vector<Tensor<1,dim> >(n_components));
            psi_scalar.resize (n_q_points);

            tmp_values.resize (n_q_points);
            tmp_gradients.resize (n_q_points);

            if (weight!=0)
              {
                if (weight->n_components>1)
                  weight->vector_value_list (fe_values.get_quadrature_points(),
                                             weight_vectors);
                else
                  {
                    weight->value_list (fe_values.get_quadrature_points(),
                                        weight_values);
                    for (unsigned int k=0; k<n_q_points; ++k)
                      weight_vectors[k] = weight_values[k];
                  }
              }
            else
              {
                for (unsigned int k=0; k<n_q_points; ++k)
                  weight_vectors[k] = 1.;
              }

	  
	    
            if (update_flags & update_values)
              {
		std::cout << "bin drin values" << std::endl;
                // first compute the exact solution
                // (vectors) at the quadrature points
                // try to do this as efficient as
                // possible by avoiding a second
                // virtual function call in case
                // the function really has only
                // one component
                if (fe_is_system)
                  exact_solution.vector_value_list (fe_values.get_quadrature_points(),
                                                    psi_values);
                else
                  {
                    exact_solution.value_list (fe_values.get_quadrature_points(),
                                               tmp_values);
                    for (unsigned int i=0; i<n_q_points; ++i)
                      psi_values[i](0) = tmp_values[i];
                  }

                // then subtract finite element
                // fe_function
                fe_values.get_function_values (fe_function, function_values);
                for (unsigned int q=0; q<n_q_points; ++q)
                  psi_values[q] -= function_values[q];
              }

	  
      
            // Do the same for gradients, if required
            if (update_flags & update_gradients)
              {
                // try to be a little clever
                // to avoid recursive virtual
                // function calls when calling
                // gradient_list for functions
                // that are really scalar
                // functions
                if (fe_is_system)
		  {
		    exact_solution.vector_gradient_list (fe_values.get_quadrature_points(),
							 psi_grads_tmp);
		    
		    for (unsigned int k=0; k<n_components; ++k)
		      for (unsigned int q=0; q<n_q_points; ++q)
			{
			  double a = alpha_eps_squared_local 
			    + psi_grads_tmp[q][k][0] * psi_grads_tmp[q][k][0]
			    + psi_grads_tmp[q][k][1] * psi_grads_tmp[q][k][1];
			  psi_grads[q][k] = std::pow(a,power_local) * psi_grads_tmp[q][k];

			}
		    
		  }
                else
                  {
                    exact_solution.gradient_list (fe_values.get_quadrature_points(),
                                                  tmp_gradients);

                    for (unsigned int i=0; i<n_q_points; ++i)
		      {

			double a = alpha_eps_squared_local 
			  + tmp_gradients[i][0] * tmp_gradients[i][0]
			  + tmp_gradients[i][1] * tmp_gradients[i][1];
			psi_grads[i][0] = std::pow(a,power_local) * tmp_gradients[i];
		      }
                  }

 
                fe_values.get_function_gradients (fe_function, function_grads);

                  for (unsigned int k=0; k<n_components; ++k)
                    for (unsigned int q=0; q<n_q_points; ++q)
		      {
			double b = alpha_eps_squared_local 
			  + function_grads[q][k][0] * function_grads[q][k][0]
			  + function_grads[q][k][1] * function_grads[q][k][1];

			psi_grads[q][k] -= std::pow(b,power_local) * function_grads[q][k];

		      }
              

	      } // end (update_flags & gradients)

	  	 
      


                // take square of integrand
                std::fill_n (psi_scalar.begin(), n_q_points, 0.0);
                for (unsigned int k=0; k<n_components; ++k)
                  for (unsigned int q=0; q<n_q_points; ++q)
                    psi_scalar[q] += (psi_grads[q][k] * psi_grads[q][k])
                                     * weight_vectors[q](k);

                // add seminorm to L_2 norm or
                // to zero
                diff += std::inner_product (psi_scalar.begin(), psi_scalar.end(),
                                            fe_values.get_JxW_values().begin(),
                                            0.0);
                diff = std::sqrt(diff);

            difference(index) = diff;
	  }
        else
          // the cell is a ghost cell
          // or is artificial. write
          // a zero into the
          // corresponding value of
          // the returned vector
          difference(index) = 0;


}
      






  // Like several of the functions above, this is almost exactly a copy of of
  // the corresponding function in step-6. The only difference is the call to
  // <code>assemble_multigrid</code> that takes care of forming the matrices
  // on every level that we need in the multigrid method.
  template <int dim>
  void LaplaceProblem<dim>::run ()
  {
    time_Newton_global_current_mesh = 0.0;
    
    // Defining test cases
    // These correspond to example 1-3 in the paper TouWi17, SISC, 2017
    test_case = 1;
    
    if (test_case == 1)
      set_runtime_parameters_example_1 ();
    else if (test_case == 2)
      set_runtime_parameters_example_2 ();
    else if (test_case == 3)
      set_runtime_parameters_example_3 ();

    setup_system();

    //ConstraintMatrix constraints;
    AffineConstraints<double> constraints;
    constraints.close();
    
    //std::vector<bool> component_mask (dim+1, true);
    VectorTools::project (mg_dof_handler,
			  constraints,
			  QGauss<dim>(degree+2),
			  ManufacturedSolution<dim>(test_case,gamma,omega),
			  solution
			  );

    // Terminal output
    unsigned int fe_degree = 1;
    std::cout << "Info: FE degree\teps\tp\tTOL (LinSolve)\tTOL(Newton)" << std::endl;
    std::cout << "Info: " 
	      << fe_degree << "\t\t"  
	      << alpha_eps << "\t" 
	      << power_p << "\t"
	      << lower_bound_linear_solver << "\t\t"
	      << lower_bound_newton_residual 
	      << std::endl;
    std::cout << "Info:  " <<  std::endl;
    //std::cout << "Info: Cells\tDoFs\th\t\tF-norm err\tConvRate\tMin/MaxLinIter\tNewton iter" << std::endl;
    std::cout << "Info: Cells\tDoFs\th\t\tMin/MaxLinIter\tNewton iter\tCPU time[s]" << std::endl;


    for (unsigned int cycle=0; cycle<max_no_refinement_cycles; ++cycle)
      {
	std::cout << "\n===============================" 
		  << "=====================================" 
		  << std::endl; 
	std::cout << "Refinment cycle " << cycle << ':' << std::endl;

	  // Take as initial Newton guess the 
	  // interpolated solution from the previous mesh
	  if (cycle > 0)
	    {
//	      VectorTools::project (mg_dof_handler,
//			  constraints,
//			  QGauss<dim>(degree+2),
//			  ManufacturedSolution<dim>(test_case,gamma,omega),
//			  solution
//			  );

	      Vector<double> tmp_solution;
	      tmp_solution = solution;
	      
	      SolutionTransfer<dim,Vector<double> > solution_transfer (mg_dof_handler);
	      solution_transfer.prepare_for_coarsening_and_refinement(tmp_solution);

	      triangulation.refine_global (1);
 
	      std::cout << "   Number of active cells:       "
			<< triangulation.n_active_cells()
			<< std::endl;
	      setup_system();
	      solution_transfer.interpolate(tmp_solution, solution); 



	    }



        std::cout << "   Number of degrees of freedom: "
                  << mg_dof_handler.n_dofs()
                  << " (by level: ";
        for (unsigned int level=0; level<triangulation.n_levels(); ++level)
          std::cout << mg_dof_handler.n_dofs(level)
                    << (level == triangulation.n_levels()-1
                        ? ")" : ", ");
        std::cout << std::endl;

	// Playing with different eps on different meshes
	if (cycle > 0)
	{
	  //alpha_eps = 5.0 * 22.627e-1 * min_cell_diameter; //0.1;// * alpha_eps;
	  //alpha_eps = 0.5 * alpha_eps;
	  std::cout << "   eps: " << alpha_eps << std::endl;
	}

	std::cout << "---------------------------------" 
		  << "-----------------------------------" 
		  << std::endl; 
        std::cout << std::endl;


	// Solving the nonlinear problem
	// At each Neweton step, the linear equation
	// systems are solved with CG, which is 
	// preconditioned with geometric multigrid
	delta_fixed_point_newton = 1.0; // TODO !!!! Should be 1
	newton_iteration ();  

	// Compute functional values, e.g., here error norms
	std::cout << std::endl;
	compute_functional_values(); 

	// Write vtk output
        output_results (cycle);
      }
  }



//
// This is again the same function as in allmost all other deal.II tutorial steps
int main ()
{
  try
    {
      using namespace dealii;

      deallog.depth_console (0);

      // The "<2>" is the spatial dimension of the problem
      // The "(1)" is the polynomial degree
      LaplaceProblem<2> p_laplace_problem(1);
      p_laplace_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
