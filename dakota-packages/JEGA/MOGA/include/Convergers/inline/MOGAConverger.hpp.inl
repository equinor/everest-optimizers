/*
================================================================================
    PROJECT:

        John Eddy's Genetic Algorithms (JEGA)

    CONTENTS:

        Inline methods of class MOGAConverger.

    NOTES:

        See notes of MOGAConverger.hpp.

    PROGRAMMERS:

        John Eddy (jpeddy@sandia.gov) (JE)

    ORGANIZATION:

        Sandia National Laboratories

    COPYRIGHT:

        See the LICENSE file in the top level JEGA directory.

    VERSION:

        1.0.0

    CHANGES:

        Mon Jul 07 09:24:18 2003 - Original Version (JE)

================================================================================
*/




/*
================================================================================
Document This File
================================================================================
*/
/** \file
 * \brief Contains the inline methods of the MOGAConverger class.
 */


/*
================================================================================
Includes
================================================================================
*/







/*
================================================================================
Begin Namespace
================================================================================
*/
namespace JEGA {
    namespace Algorithms {








/*
================================================================================
Inline Mutators
================================================================================
*/








/*
================================================================================
Inline Accessors
================================================================================
*/








/*
================================================================================
Inline Public Methods
================================================================================
*/








/*
================================================================================
Inline Subclass Visible Methods
================================================================================
*/

inline
void
MOGAConverger::UpdateParetoExtremes(
    const eddy::utilities::extremes<obj_val_t>& newParExtremes
    )
{
    this->_prevParExtremes = newParExtremes;
}

inline
void
MOGAConverger::UpdatePopulationExtremes(
    const eddy::utilities::extremes<obj_val_t>& newPopExtremes
    )
{
    this->_prevPopExtremes = newPopExtremes;
}







/*
================================================================================
Inline Private Methods
================================================================================
*/


#ifdef JEGA_MESSAGE_BOARD

inline
const JEGA::Utilities::MessageInfo&
MOGAConverger::DEN_MSG_INFO(
    )
{
    static const JEGA::Utilities::MessageInfo msgInfo(
        JEGA::Utilities::MessageBoard::MessageIdentifier(
            0x0, "metric_tracker", "density"
            )
        );

    return msgInfo;
}


inline
const JEGA::Utilities::MessageInfo&
MOGAConverger::EXP_MSG_INFO(
    )
{
    static const JEGA::Utilities::MessageInfo msgInfo(
        JEGA::Utilities::MessageBoard::MessageIdentifier(
            0x0, "metric_tracker", "expanse"
            )
        );

    return msgInfo;
}

inline
const JEGA::Utilities::MessageInfo&
MOGAConverger::DEP_MSG_INFO(
    )
{
    static const JEGA::Utilities::MessageInfo msgInfo(
        JEGA::Utilities::MessageBoard::MessageIdentifier(
            0x0, "metric_tracker", "depth"
            )
    );

    return msgInfo;
}

#endif

/*
================================================================================
Inline Structors
================================================================================
*/








/*
================================================================================
End Namespace
================================================================================
*/
    } // namespace Algorithms
} // namespace JEGA
