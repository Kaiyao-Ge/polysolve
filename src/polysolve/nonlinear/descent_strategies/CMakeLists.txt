set(SOURCES
	DescentStrategy.hpp
	LBFGS.hpp
	LBFGS.cpp
	BFGS.cpp
	BFGS.hpp
	GradientDescent.cpp
	GradientDescent.hpp
	ADAM.cpp
	ADAM.hpp
	Newton.hpp
	Newton.cpp
	NonsmoothNewton.hpp
	NonsmoothNewton.cpp
	NonsmoothNewtonBase.hpp
	NonsmoothNewtonBase.cpp
	BlindNewton.hpp
	BlindNewton.cpp
	nonsmooth/BasisMatrix.hpp
	nonsmooth/BasisMatrix.cpp
	nonsmooth/SubspaceModel.hpp
	nonsmooth/SubspaceModel.cpp
	nonsmooth/ReducedMatrix.hpp
	nonsmooth/ReducedMatrix.cpp
	nonsmooth/PathRunner.hpp
	nonsmooth/PathRunner.cpp
	ParallelKernels.hpp
	ParallelKernels.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(box_constraints)
