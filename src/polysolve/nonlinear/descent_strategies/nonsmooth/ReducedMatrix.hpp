#pragma once

#include <polysolve/nonlinear/Problem.hpp>

#include <memory>
#include <string>
#include <vector>

namespace polysolve::nonlinear
{
    /// @brief Nonsmooth term at a trial point y = x + p
    struct NonsmoothEval
    {
        Eigen::VectorXd y; ///< trial point
        double E = 0;      ///< value
        bool deriv = false;
        Eigen::VectorXd g;                ///< gradient
        StiffnessMatrix K;                ///< unprojected Hessian (global path)
        std::vector<HessianBlock> blocks; ///< unprojected Hessian blocks (local path)
        bool local = false;
    };

    /// @brief The matrix A = G + Q^T K Q of the reduced model, K the unprojected Hessian of the nonsmooth term at a trial
    /// point. Two ways to form Q^T K Q from the local blocks:
    /// per_block: block by block, S += L_b^T H_b L_b with L_b the rows of Q of the block;
    /// merged: the blocks of the first point of an outer iteration fix a sparse structure K_R over the rows they touch; at
    /// every point the values of the blocks that have the same index lists are written to their positions, and
    /// A = G + Q_R^T (K_R Q_R) (one sparse-dense product and one dense product); blocks that are not in the structure (new at
    /// a later point, or from units made exact in the path) are added block by block.
    /// The two round differently (about 1e-13 relative). Mode auto takes the merged form if the first point of an outer
    /// iteration has at least merged_min_blocks blocks.
    class ReducedMatrix
    {
    public:
        using Mat = Eigen::MatrixXd;

        /// @param mode per_block, auto or merged
        explicit ReducedMatrix(const std::string &mode = "per_block", const int merged_min_blocks = 2000)
            : mode_(mode), min_blocks_(merged_min_blocks)
        {
        }

        using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        /// @brief Rows of Q are read from this row-major copy (same values, contiguous rows); nullptr: from Q
        void set_row_major_q(const RowMat *Qr) { qr_ = Qr; }

        /// @brief A new outer iteration: the structure of the last one is forgotten, the next build is its first point
        void begin_outer();

        /// @brief A (before the symmetrization) from the k x k part G of the basis Gram matrix and the first k columns of Q
        void build(const Mat &Q, const Mat &G, const NonsmoothEval &c, const int k, Mat &A);

        /// @brief True if the current outer iteration uses the merged form (after its first build)
        bool merged() const { return merged_; }
        /// @brief The structure of the merged form: its number of stored entries and the rows (reduced dofs) it covers, in the
        /// order they were first met; empty without a structure
        long structure_nnz() const;
        std::vector<int> structure_rows() const;

    private:
        struct Structure;
        /// A without the structure: block by block (also the blocks of the merged form that are not in the structure)
        void add_blocks(const Mat &Q, const NonsmoothEval &c, const int k, Mat &S, const Eigen::Index b0, const Eigen::Index b1) const;
        void make_structure(const NonsmoothEval &c, const Eigen::Index n);
        void build_merged(const Mat &Q, const Mat &G, const NonsmoothEval &c, const int k, Mat &A);

        /// row r of the first k columns of Q into dst
        template <typename Dst>
        void gather(const Mat &Q, const Eigen::Index r, const int k, Dst &&dst) const
        {
            if (qr_ != nullptr)
                dst = qr_->block(r, 0, 1, k);
            else
                dst = Q.block(r, 0, 1, k);
        }

        const RowMat *qr_ = nullptr;
        std::string mode_;
        int min_blocks_;
        bool first_ = true;   ///< the next build is the first point of an outer iteration
        bool merged_ = false;
        std::shared_ptr<Structure> t_;
    };
} // namespace polysolve::nonlinear
