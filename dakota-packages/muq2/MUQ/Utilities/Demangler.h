#ifndef DEMANGLER_H
#define DEMANGLER_H

#include <string>
#include <typeinfo>

namespace muq{
namespace Utilities{

  /** Demangles a string returned by type_info::name. */
  std::string demangle(const char* name);


  template<typename PointerType>
  std::string GetTypeName(PointerType const& ptr){
    if(ptr.get()){
      auto& r = *ptr.get();
      return demangle(typeid(r).name());
    }else{
      return "";
    }
  }

}
}


#endif
