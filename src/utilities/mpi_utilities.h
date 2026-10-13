#ifndef MPI_UTILITIES_H_
#define MPI_UTILITIES_H_

#include <vector>
#include "mpi.h"


namespace zerork 
{
namespace utilities
{

class mpi_counter {
  public:
    mpi_counter(MPI_Comm comm, int host_rank);
    virtual ~mpi_counter();
    int increment(int amount);
    void print();
    void reset();
  private:
    MPI_Comm comm_;
    MPI_Win win_;
    int  host_rank_ ;
    std::vector<int> data_;
    int rank_, size_;
};

}
}

#endif
