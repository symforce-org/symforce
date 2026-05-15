
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/constants.h>

namespace SymEngine
{

vec_basic generate_fdiff_weights_vector(const vec_basic &grid,
                                        const unsigned max_deriv,
                                        const RCP<const Basic> around)
{
    // Parameters
    // ----------
    // grid: grid point locations
    // max_deriv: highest derivative.
    // around: location where approximations are to be accurate
    //
    // Returns
    // -------
    // weights[grid_index, deriv_order]: weights of order 0 to max_deriv (column
    // major order)
    //
    // References
    // ----------
    // Generation of Finite Difference Formulas on Arbitrarily Spaced Grids
    //     Bengt Fornberg, Mathematics of compuation, 51, 184, 1988, 699-706
    //
    const unsigned len_g = numeric_cast<unsigned>(grid.size());
    const unsigned len_w = len_g * (max_deriv + 1);
    RCP<const Basic> c1, c4, c5;
    c1 = one;
    c4 = sub(grid[0], around);
    vec_basic weights(len_w);
    weights[0] = one;
    for (unsigned idx = 1; idx < len_w; ++idx)
        weights[idx] = zero; // clear weights
    for (unsigned i = 1; i < len_g; ++i) {
        const int mn = (i < max_deriv) ? i : max_deriv; // min(i, max_deriv)
        RCP<const Basic> c2 = one;
        c5 = c4;
        c4 = sub(grid[i], around);
        for (unsigned j = 0; j < i; ++j) {
            const RCP<const Basic> c3 = sub(grid[i], grid[j]);
            c2 = mul(c2, c3);
            if (j == i - 1) {
                for (int k = mn; k >= 1; --k) {
                    weights[i + k * len_g]
                        = div(mul(c1, sub(mul(integer(k),
                                              weights[i - 1 + (k - 1) * len_g]),
                                          mul(c5, weights[i - 1 + k * len_g]))),
                              c2);
                }
                weights[i]
                    = mul(minus_one, div(mul(c1, mul(c5, weights[i - 1])), c2));
            }
            for (int k = mn; k >= 1; --k) {
                weights[j + k * len_g]
                    = div(sub(mul(c4, weights[j + k * len_g]),
                              mul(integer(k), weights[j + (k - 1) * len_g])),
                          c3);
            }
            weights[j] = div(mul(c4, weights[j]), c3);
        }
        c1 = c2;
    }
    return weights;
}
} // namespace SymEngine
