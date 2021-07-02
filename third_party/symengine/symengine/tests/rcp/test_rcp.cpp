#include "catch.hpp"

#include <symengine/symengine_rcp.h>

using SymEngine::RCP;
using SymEngine::make_rcp;
using SymEngine::Ptr;
using SymEngine::null;
using SymEngine::EnableRCPFromThis;

// This is the canonical use of EnableRCPFromThis:

class Mesh : public EnableRCPFromThis<Mesh>
{
public:
    int x, y;
};

TEST_CASE("Test make_rcp", "[rcp]")
{
    RCP<Mesh> m = make_rcp<Mesh>();
    Ptr<Mesh> p = m.ptr();
    REQUIRE(not(m == null));
    REQUIRE(p->use_count() == 1);
    RCP<Mesh> m2 = m;
    REQUIRE(p->use_count() == 2);
    RCP<Mesh> m3 = m2;
    REQUIRE(p->use_count() == 3);
}

void f(Mesh &m)
{
    REQUIRE(m.use_count() == 1);
    // rcp_from_this() gives up non const version of RCP<Mesh> because 'm' is
    // not const
    RCP<Mesh> m2 = m.rcp_from_this();
    REQUIRE(m.use_count() == 2);
    m2->x = 6;
}

void f_const(const Mesh &m)
{
    REQUIRE(m.use_count() == 1);
    // rcp_from_this() gives up const version of RCP<Mesh> because 'm' is const
    RCP<const Mesh> m2 = m.rcp_from_this();
    REQUIRE(m.use_count() == 2);
}

TEST_CASE("Test rcp_from_this", "[rcp]")
{
    RCP<Mesh> m = make_rcp<Mesh>();
    REQUIRE(m->use_count() == 1);
    m->x = 5;
    REQUIRE(m->x == 5);
    f(*m);
    REQUIRE(m->use_count() == 1);
    REQUIRE(m->x == 6);

    f_const(*m);
    REQUIRE(m->use_count() == 1);
}

TEST_CASE("Test rcp_from_this const", "[rcp]")
{
    RCP<const Mesh> m = make_rcp<const Mesh>();
    REQUIRE(m->use_count() == 1);
    f_const(*m);
    REQUIRE(m->use_count() == 1);
}

// This is not a canonical way how to use EnableRCPFromThis, since we use
// 'const Mesh2' for the internal weak pointer, so we can only get
// 'RCP<const Mesh2>' out of rcp_from_this(). But it is legitimate code, so we
// test it as well.

class Mesh2 : public EnableRCPFromThis<const Mesh2>
{
public:
    int x, y;
};

void f2_const(const Mesh2 &m)
{
    REQUIRE(m.use_count() == 1);
    // rcp_from_this() gives up const version of RCP<Mesh> because 'm' is const
    RCP<const Mesh2> m2 = m.rcp_from_this();
    REQUIRE(m.use_count() == 2);
}

void f2_hybrid(Mesh2 &m)
{
    REQUIRE(m.use_count() == 1);
    // rcp_from_this() gives up const version of RCP<Mesh> even though 'm' is
    // not const, because the internal pointer inside Mesh2 is const.
    RCP<const Mesh2> m2 = m.rcp_from_this();
    REQUIRE(m.use_count() == 2);
}

TEST_CASE("Test rcp_from_this const 2", "[rcp]")
{
    RCP<const Mesh2> m = make_rcp<const Mesh2>();
    REQUIRE(m->use_count() == 1);
    f2_const(*m);
    REQUIRE(m->use_count() == 1);

    RCP<Mesh2> m2 = make_rcp<Mesh2>();
    REQUIRE(m2->use_count() == 1);
    f2_const(*m2);
    REQUIRE(m2->use_count() == 1);
    f2_hybrid(*m2);
    REQUIRE(m2->use_count() == 1);
}
