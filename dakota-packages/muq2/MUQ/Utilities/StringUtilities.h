#ifndef MUQSTRINGUTILITIES_H
#define MUQSTRINGUTILITIES_H

#include <string>
#include <vector>

namespace muq {
    namespace Utilities{

        namespace StringUtilities{

            /// Split a string into parts based on a particular character delimiter.  Also Strips whitespace from parts
            std::vector<std::string> Split(std::string str, char delim = ',');

            /// Strip the whitespace off the beginning and end of a string
            std::string Strip(std::string str);

            /// Combine a vector of strings with a specified delimiter.  The opposite of Split
            std::string Combine(std::vector<std::string> strs, char delim = ',');

        }
    }
}


#endif  //#ifndef MUQSTRINGUTILITIES_H
