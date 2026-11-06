######################################################################################
# Idefix MHD astrophysical code
#
# Source file src/utils/spectral/CMakeLists.txt
#
# Last modified : 11/2026
#
# Copyright(C) by :
# - Geoffroy Lesur <geoffroy.lesur@univ-grenoble-alpes.fr> (IPAG/UGA/CNRS - 2025)
# - Steven Rendon Restrepo <srendon@aip.de> (AIP - 2026)
# and other code contributors
#
# Licensed under CeCILL 2.1 License, see COPYING for more information
######################################################################################

target_sources(idefix
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/heffteFft.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/heffteFft.cpp
  )
