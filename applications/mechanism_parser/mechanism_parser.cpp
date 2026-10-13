#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <string>
#include <map>

#include "zerork/mechanism.h"
#include "utilities/file_utilities.h"

#include "MechanismParserIFP.h"

using zerork::getHighResolutionTime;

int main(int argc, char *argv[])
{
  //double clock_time = getHighResolutionTime();
  //double setup_time, loop_time, sensanal_time;

  if(argc < 2) {
    printf("# ERROR: Incorrect command line usage.\n");
    printf("#        use %s <input parameters>\n",argv[0]);
    exit(1);
  }

  if(!zerork::utilities::FileIsReadable(argv[1])) {
    printf("# ERROR: Input file %s is not readable\n",argv[1]);
    exit(1);
  }

  // Initialize flame params
  MechanismParserIFP inputFile(argv[1]);

  zerork::mechanism mech(inputFile.mech_file().c_str(),
                         inputFile.therm_file().c_str(),
			 inputFile.log_file().c_str());


  int n_elems = mech.getNumElements();
  printf("n_elems: %d\n", n_elems);
  std::map<std::string, double> eInfo = mech.getElementInfo();
  for(const auto& pair: eInfo) {
    printf("%s: %g\n", pair.first.c_str(), pair.second);
  }

  int n_species = mech.getNumSpecies();
  printf("n_species: %d\n", n_species);
  std::map<std::string, std::map<std::string, int>> seInfo = mech.getSpeciesElementInfo();
  for(const auto& pair: seInfo) {
    std::string name = pair.first;
    std::map<std::string, int> speciesElems = pair.second;
    printf("%s : {", name.c_str());
    double mw = 0;
    for(const auto& epair : speciesElems ) {
      std::string eName = epair.first;
      int eAmount = epair.second;
      mw += eInfo[eName];
      printf("%s: %d, ", eName.c_str(), eAmount);
    }
    printf(", %g}\n",mw);
  }

  int n_reactions = mech.getNumReactions();
  printf("n_reactions: %d\n", n_reactions);
  for(int i = 0; i < n_reactions; ++i) {
    printf("%s\n", mech.getReactionName(i));
  }

  return 0;
}


