// @HEADER
// ***********************************************************************
//
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
// @HEADER

#include "Teuchos_RCPNode.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_Exceptions.hpp"


// Defined this to see tracing of RCPNodes created and destroyed
//#define RCP_NODE_DEBUG_TRACE_PRINT


//
// Internal implementatation stuff
//


namespace {


//
// Local implementation types
//


struct RCPNodeInfo {
  RCPNodeInfo() : nodePtr(0) {}
  RCPNodeInfo(const std::string &info_in, Teuchos::RCPNode* nodePtr_in)
    : info(info_in), nodePtr(nodePtr_in)
    {}
  std::string info;
  Teuchos::RCPNode* nodePtr;
};


typedef std::pair<const void*, RCPNodeInfo> VoidPtrNodeRCPInfoPair_t;


typedef std::multimap<const void*, RCPNodeInfo> rcp_node_list_t;


class RCPNodeInfoListPred {
public:
  bool operator()(const rcp_node_list_t::value_type &v1,
    const rcp_node_list_t::value_type &v2
    ) const
    {
#ifdef TEUCHOS_DEBUG
      return v1.second.nodePtr->insertion_number() < v2.second.nodePtr->insertion_number();
#else
      return v1.first < v2.first;
#endif
    }
};


//
// Local static functions returning references to local static objects to
// ensure objects are initilaized.
//
// Technically speaking, the static functions on RCPNodeTracer that use this
// data might be called from other translation units in pre-main code before
// this translation unit gets initialized.  By using functions returning
// references to local static varible trick, we ensure that these objects are
// always initialized before they are used, no matter what.
//
// These could have been static functions on RCPNodeTracer but the advantage
// of defining these functions this way is that you can add and remove
// functions without affecting the *.hpp file and therefore avoid
// recompilation (and even relinking with shared libraries).
//


rcp_node_list_t*& rcp_node_list()
{
  static rcp_node_list_t *s_rcp_node_list = 0;
  // Here we must let the ActiveRCPNodesSetup constructor and destructor handle
  // the creation and destruction of this map object.  This will ensure that
  // this map object will be valid when any global/static RCP objects are
  // destroyed!  Note that this object will get created and destroyed
  // reguardless if whether we are tracing RCPNodes or not.  This just makes our
  // life simpler.  NOTE: This list will always get allocated no mater if
  // TEUCHOS_DEBUG is defined or node traceing is enabled or not.
  return s_rcp_node_list;
}


bool& loc_isTracingActiveRCPNodes()
{
  static bool s_loc_isTracingActiveRCPNodes =
#if defined(TEUCHOS_DEBUG) && defined(HAVE_TEUCHOS_DEBUG_RCP_NODE_TRACING)
    true
#else
    false
#endif
    ;
  return s_loc_isTracingActiveRCPNodes;
}


Teuchos::RCPNodeTracer::RCPNodeStatistics& loc_rcpNodeStatistics()
{
  static Teuchos::RCPNodeTracer::RCPNodeStatistics s_loc_rcpNodeStatistics;
  return s_loc_rcpNodeStatistics;
}


bool& loc_printRCPNodeStatisticsOnExit()
{
  static bool s_loc_printRCPNodeStatisticsOnExit = false;
  return s_loc_printRCPNodeStatisticsOnExit;
}


//
// Other helper functions
//

// This function returns the const void* value that is used as the key to look
// up an RCPNode object that has been stored.  If the RCPNode is holding a
// non-null reference, then we use that object address as the key.  That way,
// we can detect if a user trys to create a new owning RCPNode to the same
// object.  If the RCPNode has an null internal object pointer, then we will
// use the RCPNode's address itself.  In this case, we want to check and see
// that all RCPNodes that get created get destroyed correctly.
const void* get_map_key_void_ptr(const Teuchos::RCPNode* rcp_node)
{
  TEUCHOS_ASSERT(rcp_node);
#ifdef TEUCHOS_DEBUG
  const void* base_obj_map_key_void_ptr = rcp_node->get_base_obj_map_key_void_ptr();
  if (base_obj_map_key_void_ptr)
    return base_obj_map_key_void_ptr;
#endif
  return rcp_node;
}


std::string convertRCPNodeToString(const Teuchos::RCPNode* rcp_node)
{
  std::ostringstream oss;
  oss
    << "RCPNode {address="
    << rcp_node
#ifdef TEUCHOS_DEBUG
    << ", base_obj_map_key_void_ptr=" << rcp_node->get_base_obj_map_key_void_ptr()
#endif
    << ", base_obj_type_name=" << rcp_node->get_base_obj_type_name()
    << ", map_key_void_ptr=" << get_map_key_void_ptr(rcp_node)
    << ", has_ownership=" << rcp_node->has_ownership()
#ifdef TEUCHOS_DEBUG
    << ", insertionNumber="<< rcp_node->insertion_number()
#endif
    << "}";
  return oss.str();
}


} // namespace


namespace Teuchos {


//
// RCPNode
//


void RCPNode::set_extra_data(
  const any &extra_data, const std::string& name
  ,EPrePostDestruction destroy_when
  ,bool force_unique
  )
{
  if(extra_data_map_==NULL) {
    extra_data_map_ = new extra_data_map_t;
  }
  const std::string type_and_name( extra_data.typeName() + std::string(":") + name );
  extra_data_map_t::iterator itr = extra_data_map_->find(type_and_name);
#ifdef TEUCHOS_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(
    (itr != extra_data_map_->end() && force_unique), std::invalid_argument
    ,"Error, the type:name pair \'" << type_and_name
    << "\' already exists and force_unique==true!" );
#endif
  if (itr != extra_data_map_->end()) {
    // Change existing extra data
    itr->second = extra_data_entry_t(extra_data,destroy_when);
  }
  else {
    // Insert new extra data
    (*extra_data_map_)[type_and_name] =
      extra_data_entry_t(extra_data,destroy_when);
  }
}


any& RCPNode::get_extra_data( const std::string& type_name, const std::string& name )
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(
    extra_data_map_==NULL, std::invalid_argument
    ,"Error, no extra data has been set yet!" );
#endif
  any *extra_data = get_optional_extra_data(type_name,name);
#ifdef TEUCHOS_DEBUG
  if (!extra_data) {
    const std::string type_and_name( type_name + std::string(":") + name );
    TEUCHOS_TEST_FOR_EXCEPTION(
      extra_data == NULL, std::invalid_argument
      ,"Error, the type:name pair \'" << type_and_name << "\' is not found!" );
  }
#endif
  return *extra_data;
}


any* RCPNode::get_optional_extra_data( const std::string& type_name,
  const std::string& name )
{
  if( extra_data_map_ == NULL ) return NULL;
  const std::string type_and_name( type_name + std::string(":") + name );
  extra_data_map_t::iterator itr = extra_data_map_->find(type_and_name);
  if(itr != extra_data_map_->end())
    return &(*itr).second.extra_data;
  return NULL;
}


void RCPNode::impl_pre_delete_extra_data()
{
  for(
    extra_data_map_t::iterator itr = extra_data_map_->begin();
    itr != extra_data_map_->end();
    ++itr
    )
  {
    extra_data_map_t::value_type &entry = *itr;
    if(entry.second.destroy_when == PRE_DESTROY)
      entry.second.extra_data = any();
  }
}


//
// RCPNodeTracer
//


// General user functions


bool RCPNodeTracer::isTracingActiveRCPNodes()
{
  return loc_isTracingActiveRCPNodes();
}


#if defined(TEUCHOS_DEBUG) && !defined(HAVE_TEUCHOS_DEBUG_RCP_NODE_TRACING)
void RCPNodeTracer::setTracingActiveRCPNodes(bool tracingActiveNodes)
{
  loc_isTracingActiveRCPNodes() = tracingActiveNodes;
}
#endif


int RCPNodeTracer::numActiveRCPNodes()
{
  // This list always exists, no matter debug or not so just access it.
  TEUCHOS_TEST_FOR_EXCEPT(0==rcp_node_list());
  return rcp_node_list()->size();
  return 0;
}


RCPNodeTracer::RCPNodeStatistics
RCPNodeTracer::getRCPNodeStatistics()
{
  return loc_rcpNodeStatistics();
}

void RCPNodeTracer::printRCPNodeStatistics(
    const RCPNodeStatistics& rcpNodeStatistics, std::ostream &out)
{
  out
    << "\n***"
    << "\n*** RCPNode Tracing statistics:"
    << "\n**\n"
    << "\n    maxNumRCPNodes             = "<<rcpNodeStatistics.maxNumRCPNodes
    << "\n    totalNumRCPNodeAllocations = "<<rcpNodeStatistics.totalNumRCPNodeAllocations
    << "\n    totalNumRCPNodeDeletions   = "<<rcpNodeStatistics.totalNumRCPNodeDeletions
    << "\n";
}


void RCPNodeTracer::setPrintRCPNodeStatisticsOnExit(
  bool printRCPNodeStatisticsOnExit)
{
  loc_printRCPNodeStatisticsOnExit() = printRCPNodeStatisticsOnExit;
}


bool RCPNodeTracer::getPrintRCPNodeStatisticsOnExit()
{
  return loc_printRCPNodeStatisticsOnExit();
}


void RCPNodeTracer::printActiveRCPNodes(std::ostream &out)
{
#ifdef TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODE_TRACE
  out
    << "\nCalled printActiveRCPNodes() :"
    << " rcp_node_list.size() = " << rcp_node_list().size() << "\n";
#endif // TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODE_TRACE
  if (loc_isTracingActiveRCPNodes()) {
    TEUCHOS_TEST_FOR_EXCEPT(0==rcp_node_list());
    if (rcp_node_list()->size() > 0) {
      out << getActiveRCPNodeHeaderString();
      // Create a sorted-by-insertionNumber list
      // NOTE: You have to use std::vector and *not* Teuchos::Array rcp here
      // because this called at the very end and uses RCPNode itself in a
      // debug-mode build.
      typedef std::vector<VoidPtrNodeRCPInfoPair_t> rcp_node_vec_t;
      rcp_node_vec_t rcp_node_vec(rcp_node_list()->begin(), rcp_node_list()->end());
      std::sort(rcp_node_vec.begin(), rcp_node_vec.end(), RCPNodeInfoListPred());
      // Print the RCPNode objects sorted by insertion number
      typedef rcp_node_vec_t::const_iterator itr_t;
      int i = 0;
      for ( itr_t itr = rcp_node_vec.begin(); itr != rcp_node_vec.end(); ++itr ) {
        const rcp_node_list_t::value_type &entry = *itr;
        TEUCHOS_ASSERT(entry.second.nodePtr);
        out
          << "\n"
          << std::setw(3) << std::right << i << std::left
          << ": RCPNode (map_key_void_ptr=" << entry.first << ")\n"
          << "       Information = " << entry.second.info << "\n"
          << "       RCPNode address = " << entry.second.nodePtr << "\n"
#ifdef TEUCHOS_DEBUG
          << "       insertionNumber = " << entry.second.nodePtr->insertion_number()
#endif
          ;
        ++i;
      }
      out << "\n\n"
          << getCommonDebugNotesString();
    }
  }
}


// Internal implementation functions


void RCPNodeTracer::addNewRCPNode( RCPNode* rcp_node, const std::string &info )
{

  // Used to allow unique identification of rcp_node to allow setting breakpoints
  static int insertionNumber = 0;

  // Set the insertion number right away in case an exception gets thrown so
  // that you can set a break point to debug this.
#ifdef TEUCHOS_DEBUG
  rcp_node->set_insertion_number(insertionNumber);
#endif

  if (loc_isTracingActiveRCPNodes()) {

    // Print the node we are adding if configured to do so.  We have to send
    // to std::cerr to make sure that this gets printed.
#ifdef RCP_NODE_DEBUG_TRACE_PRINT
    std::cerr
      << "RCPNodeTracer::addNewRCPNode(...): Adding "
      << convertRCPNodeToString(rcp_node) << " ...\n";
#endif

    TEUCHOS_TEST_FOR_EXCEPT(0==rcp_node_list());

    const void * const map_key_void_ptr = get_map_key_void_ptr(rcp_node);

    // See if the rcp_node or its object has already been added.
    typedef rcp_node_list_t::iterator itr_t;
    typedef std::pair<itr_t, itr_t> itr_itr_t;
    const itr_itr_t itr_itr = rcp_node_list()->equal_range(map_key_void_ptr);
    const bool rcp_node_already_exists = itr_itr.first != itr_itr.second;
    RCPNode *previous_rcp_node = 0;
    bool previous_rcp_node_has_ownership = false;
    for (itr_t itr = itr_itr.first; itr != itr_itr.second; ++itr) {
      previous_rcp_node = itr->second.nodePtr;
      if (previous_rcp_node->has_ownership()) {
        previous_rcp_node_has_ownership = true;
        break;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(
      rcp_node_already_exists && rcp_node->has_ownership() && previous_rcp_node_has_ownership,
      DuplicateOwningRCPError,
      "RCPNodeTracer::addNewRCPNode(rcp_node): Error, the client is trying to create a new\n"
      "RCPNode object to an existing managed object in another RCPNode:\n"
      "\n"
      "  New " << convertRCPNodeToString(rcp_node) << "\n"
      "\n"
      "  Existing " << convertRCPNodeToString(previous_rcp_node) << "\n"
      "\n"
      "  Number current nodes = " << rcp_node_list()->size() << "\n"
      "\n"
      "This may indicate that the user might be trying to create a weak RCP to an existing\n"
      "object but forgot make it non-ownning.  Perhaps they meant to use rcpFromRef(...)\n"
      "or an equivalent function?\n"
      "\n"
      << getCommonDebugNotesString();
      );

    // NOTE: We allow duplicate RCPNodes if the new node is non-owning.  This
    // might indicate a advanced usage of the RCP class that we want to
    // support.  The typical problem is when the programmer unknowingly
    // creates an owning RCP to an object already owned by another RCPNode.

    // Add the new RCP node keyed as described above.
    (*rcp_node_list()).insert(
      itr_itr.second,
      std::make_pair(map_key_void_ptr, RCPNodeInfo(info, rcp_node))
      );
    // NOTE: Above, if there is already an existing RCPNode with the same key
    // value, this iterator itr_itr.second will point to one after the found
    // range.  I suspect that this might also ensure that the elements are
    // sorted in natural order.

    // Update the insertion number an node tracing statistics
    ++insertionNumber;
    ++loc_rcpNodeStatistics().totalNumRCPNodeAllocations;
    loc_rcpNodeStatistics().maxNumRCPNodes =
      TEUCHOS_MAX(loc_rcpNodeStatistics().maxNumRCPNodes, numActiveRCPNodes());
  }
}


#define TEUCHOS_RCPNODE_REMOVE_RCPNODE(CONDITION, RCPNODE) \
  TEUCHOS_TEST_FOR_EXCEPTION((CONDITION), \
    std::logic_error, \
    "RCPNodeTracer::removeRCPNode(node_ptr): Error, the " \
    << convertRCPNodeToString(RCPNODE) << " is not found in the list of" \
    " active RCP nodes being traced even though all nodes should be traced." \
    "  This should not be possible and can only be an internal programming error!")


void RCPNodeTracer::removeRCPNode( RCPNode* rcp_node )
{

  // Here, we will try to remove an RCPNode reguardless if whether
  // loc_isTracingActiveRCPNodes==true or not.  This will not be a performance
  // problem and it will ensure that any RCPNode objects that are added to
  // this list will be removed and will not look like a memory leak.  In
  // non-debug mode, this function will never be called.  In debug mode, with
  // loc_isTracingActiveRCPNodes==false, the list *rcp_node_list will be empty and
  // therefore this find(...) operation should be pretty cheap (even for a bad
  // implementation of std::map).

  TEUCHOS_ASSERT(rcp_node_list());
  typedef rcp_node_list_t::iterator itr_t;
  typedef std::pair<itr_t, itr_t> itr_itr_t;

  const itr_itr_t itr_itr =
    rcp_node_list()->equal_range(get_map_key_void_ptr(rcp_node));
  const bool rcp_node_exists = itr_itr.first != itr_itr.second;

#ifdef HAVE_TEUCHOS_DEBUG_RCP_NODE_TRACING
  // If we have the macro HAVE_TEUCHOS_DEBUG_RCP_NODE_TRACING turned on a
  // compile time, then all RCPNode objects that get created will have been
  // added to this list.  In this case, we can asset that the node exists.
  TEUCHOS_RCPNODE_REMOVE_RCPNODE(!rcp_node_exists, rcp_node);
#else
  // If the macro HAVE_TEUCHOS_DEBUG_RCP_NODE_TRACING turned off, then is is
  // possible that an RCP got created before the bool
  // loc_isTracingActiveRCPNodes was turned on.  In this case, we must allow
  // for an RCP node not to have been added to this list.  In this case we
  // will just let this go!
#endif

  if (rcp_node_exists) {
#ifdef RCP_NODE_DEBUG_TRACE_PRINT
    std::cerr
      << "RCPNodeTracer::removeRCPNode(...): Removing "
      << convertRCPNodeToString(rcp_node) << " ...\n";
#endif
    bool foundRCPNode = false;
    for(itr_t itr = itr_itr.first; itr != itr_itr.second; ++itr) {
      if (itr->second.nodePtr == rcp_node) {
        rcp_node_list()->erase(itr);
        ++loc_rcpNodeStatistics().totalNumRCPNodeDeletions;
        foundRCPNode = true;
        break;
      }
    }
    // Whoops! Did not find the node!
    TEUCHOS_RCPNODE_REMOVE_RCPNODE(!foundRCPNode, rcp_node);
  }

}


RCPNode* RCPNodeTracer::getExistingRCPNodeGivenLookupKey(const void* p)
{
  typedef rcp_node_list_t::iterator itr_t;
  typedef std::pair<itr_t, itr_t> itr_itr_t;
  if (!p)
    return 0;
  const itr_itr_t itr_itr = rcp_node_list()->equal_range(p);
  for (itr_t itr = itr_itr.first; itr != itr_itr.second; ++itr) {
    RCPNode* rcpNode = itr->second.nodePtr;
    if (rcpNode->has_ownership()) {
      return rcpNode;
    }
  }
  return 0;
  // NOTE: Above, we return the first RCPNode added that has the given key
  // value.
}


std::string RCPNodeTracer::getActiveRCPNodeHeaderString()
{
  return std::string(
    "\n***"
    "\n*** Warning! The following Teuchos::RCPNode objects were created but have"
    "\n*** not been destroyed yet.  A memory checking tool may complain that these"
    "\n*** objects are not destroyed correctly."
    "\n***"
    "\n*** There can be many possible reasons that this might occur including:"
    "\n***"
    "\n***   a) The program called abort() or exit() before main() was finished."
    "\n***      All of the objects that would have been freed through destructors"
    "\n***      are not freed but some compilers (e.g. GCC) will still call the"
    "\n***      destructors on static objects (which is what causes this message"
    "\n***      to be printed)."
    "\n***"
    "\n***   b) The program is using raw new/delete to manage some objects and"
    "\n***      delete was not called correctly and the objects not deleted hold"
    "\n***      other objects through reference-counted pointers."
    "\n***"
    "\n***   c) This may be an indication that these objects may be involved in"
    "\n***      a circular dependency of reference-counted managed objects."
    "\n***\n"
    );
}


std::string RCPNodeTracer::getCommonDebugNotesString()
{
  return std::string(
    "NOTE: To debug issues, open a debugger, and set a break point in the function where\n"
    "the RCPNode object is first created to determine the context where the object first\n"
    "gets created.  Each RCPNode object is given a unique insertionNumber to allow setting\n"
    "breakpoints in the code.  For example, in GDB one can perform:\n"
    "\n"
    "1) Open the debugger (GDB) and run the program again to get updated object addresses\n"
    "\n"
    "2) Set a breakpoint in the RCPNode insertion routine when the desired RCPNode is first\n"
    "inserted.  In GDB, to break when the RCPNode with insertionNumber==3 is added, do:\n"
    "\n"
    "  (gdb) b 'Teuchos::RCPNodeTracer::addNewRCPNode( [TAB] ' [ENTER]\n"
    "  (gdb) cond 1 insertionNumber==3 [ENTER]\n"
    "\n"
    "3) Run the program in the debugger.  In GDB, do:\n"
    "\n"
    "  (gdb) run [ENTER]\n"
    "\n"
    "4) Examine the call stack when the program breaks in the function addNewRCPNode(...)\n"
    );
}


//
// ActiveRCPNodesSetup
//


ActiveRCPNodesSetup::ActiveRCPNodesSetup()
{
#ifdef TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODE_TRACE
  std::cerr << "\nCalled ActiveRCPNodesSetup::ActiveRCPNodesSetup() : count = " << count_ << "\n";
#endif // TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODE_TRACE
  if (!rcp_node_list())
    rcp_node_list() = new rcp_node_list_t;
  ++count_;
}


ActiveRCPNodesSetup::~ActiveRCPNodesSetup()
#ifdef TEUCHOS_DEBUG
    noexcept(false)
#endif
{
#ifdef TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODE_TRACE
  std::cerr << "\nCalled ActiveRCPNodesSetup::~ActiveRCPNodesSetup() : count = " << count_ << "\n";
#endif // TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODE_TRACE
  if( --count_ == 0 ) {
#ifdef TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODE_TRACE
    std::cerr << "\nPrint active nodes!\n";
#endif // TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODE_TRACE
    std::cout << std::flush;
    TEUCHOS_TEST_FOR_TERMINATION(nullptr==rcp_node_list(), "rcp_node_list() is null in ~ActiveRCPNodesSetup");
    RCPNodeTracer::RCPNodeStatistics rcpNodeStatistics =
      RCPNodeTracer::getRCPNodeStatistics();
    if (rcpNodeStatistics.maxNumRCPNodes
      && RCPNodeTracer::getPrintRCPNodeStatisticsOnExit())
    {
      RCPNodeTracer::printRCPNodeStatistics(rcpNodeStatistics, std::cout);
    }
    RCPNodeTracer::printActiveRCPNodes(std::cerr);
    delete rcp_node_list();
    rcp_node_list() = 0;
  }
}


void Teuchos::ActiveRCPNodesSetup::foo()
{
  int dummy = count_;
  ++dummy; // Avoid unused variable warning (bug 2664)
}


int Teuchos::ActiveRCPNodesSetup::count_ = 0;


//
// RCPNodeHandle
//


void RCPNodeHandle::unbindOne()
{
  if (node_) {
    // NOTE: We only deincrement the reference count after
    // we have called delete on the underlying object since
    // that call to delete may actually thrown an exception!
    if (node_->strong_count()==1 && strength()==RCP_STRONG) {
      // Delete the object (which might throw)
      node_->delete_obj();
 #ifdef TEUCHOS_DEBUG
      // We actaully also need to remove the RCPNode from the active list for
      // some specialized use cases that need to be able to create a new RCP
      // node pointing to the same memory.  What this means is that when the
      // strong count goes to zero and the referenced object is destroyed,
      // then it will not longer be picked up by any other code and instead it
      // will only be known by its remaining weak RCPNodeHandle objects in
      // order to perform debug-mode runtime checking in case a client tries
      // to access the obejct.
      local_activeRCPNodesSetup.foo(); // Make sure created!
      RCPNodeTracer::removeRCPNode(node_);
#endif
   }
    // If we get here, no exception was thrown!
    if ( (node_->strong_count() + node_->weak_count()) == 1 ) {
      // The last RCP object is going away so time to delete
      // the entire node!
      delete node_;
      node_ = 0;
      // NOTE: No need to deincrement the reference count since this is
      // the last RCP object being deleted!
    }
    else {
      // The last RCP has not gone away so just deincrement the reference
      // count.
      node_->deincr_count(strength());
    }
  }
}


} // namespace Teuchos


//
// Non-member helpers
//


void Teuchos::throw_null_ptr_error( const std::string &type_name )
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    true, NullReferenceError,
    type_name << " : You can not call operator->() or operator*()"
    <<" if getRawPtr()==0!" );
}
