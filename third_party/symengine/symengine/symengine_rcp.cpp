#include <symengine/symengine_rcp.h>

#ifdef WITH_SYMENGINE_TEUCHOS
#include <symengine/utilities/teuchos/Teuchos_RCP.hpp>
#endif

namespace SymEngine
{

#ifdef WITH_SYMENGINE_RCP

void print_stack_on_segfault()
{
#ifdef WITH_SYMENGINE_TEUCHOS
    Teuchos::print_stack_on_segfault();
#endif
}

#endif

} // namespace SymEngine
