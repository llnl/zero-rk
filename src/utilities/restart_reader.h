#ifndef RESTART_WRITER_H
#define RESTART_WRITER_H


#ifdef ZERORK_MPI
#include <mpi.h>
#endif
#include <fstream>
#include <assert.h>
#include <string>
#include <vector>

namespace zerork {
namespace utilities {


enum RestartReaderStatus { READER_OK,
	READER_BAD_FILE,
	READER_MISSING_FIELD,
	READER_GRID_MISMATCH
};

class RestartReader {

public:
  RestartReader(std::string filename,
		int num_points
#ifdef ZERORK_MPI
                ,
                MPI_Comm comm
#endif
                )
      : time_(0), rank_(0), nranks_(1), num_points_local_(num_points),
	num_fields_(0), num_points_file_(0), status_(READER_OK)
#ifdef ZERORK_MPI
        ,
        comm_(comm)
#endif
  {
    {
      std::ifstream checkfile(filename.c_str());
      if(!checkfile.good()) {
        status_ = READER_BAD_FILE;
        return;
      }
    }
#ifdef ZERORK_MPI
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nranks_);

    MPI_File_open(comm_, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &restart_file_);
    MPI_File_read_all(restart_file_, &num_points_file_, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_read_all(restart_file_, &num_fields_, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_read_all(restart_file_, &time_, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);

    rank_offsets_.assign(nranks_, 0);
    std::vector<int> rank_points(nranks_, 0);
    rank_offsets_[rank_] = num_points_local_;
    MPI_Allreduce(rank_offsets_.data(), rank_points.data(), nranks_, MPI_INT,
                  MPI_MAX, comm_);
    rank_offsets_[rank_] = 0;
    global_size_ = 0;
    for (int i = 0; i < nranks_; ++i) {
      global_size_ += rank_points[i];
      if (i > 0) {
        rank_offsets_[i] = rank_offsets_[i - 1] + rank_points[i - 1];
      }
    }
#else
    global_size_ = num_points_local_;
    restart_file_.open(filename.c_str(), std::ios::in | std::ios::binary);
    restart_file_.read((char*)&num_points_file_, sizeof(int));
    restart_file_.read((char*)&num_fields_, sizeof(int));
    restart_file_.read((char*)&time_, sizeof(double));
#endif
    if(global_size_ != num_points_file_) {
      status_ = READER_GRID_MISMATCH;
      close_file();
      return;
    }

    header_size_ =
        2 * sizeof(int) + sizeof(double) + num_fields_ * sizeof(char) * 64;

    field_names_.resize(num_fields_);
    for(int j=0; j<num_fields_; ++j) {
      char buf[64];
#ifdef ZERORK_MPI
      MPI_File_read_all(restart_file_, &buf, 64, MPI_CHAR, MPI_STATUS_IGNORE);
#else
      restart_file_.read((char*)&buf, 64*sizeof(char));
#endif
      field_names_[j] = std::string(buf);
    }
  }

  ~RestartReader() {
    if(status_ = READER_OK) {
      close_file();
    }
  }

  void ReadField(const std::string field_name, std::vector<double> &field, RestartReaderStatus* status) {
    if(status_ != READER_OK) {
      *status = status_;
      return;
    }
    assert(field.size() == num_points_local_);
    assert(global_size_ == num_points_file_);

    int field_index = -1;
    //N.B. this is not currently a named field but stashed at the end of the restart file
    if(field_name == "mass_flux") {
        field_index = num_fields_;
    }
    for(int i = 0; i < num_fields_; ++i) {
      if(field_name == field_names_[i]) {
        field_index = i;
        break;
      }
    }
    if(field_index == -1) {
      *status = READER_MISSING_FIELD;
      return;
    }

    int offset = header_size_;
    offset += field_index * global_size_ * sizeof(double);
#ifdef ZERORK_MPI
    offset += rank_offsets_[rank_] * sizeof(double);
    MPI_File_set_view(restart_file_, offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read(restart_file_, field.data(), num_points_local_, MPI_DOUBLE, MPI_STATUS_IGNORE);
#else
    restart_file_.seekg(offset);
    restart_file_.read((char *)field.data(),
                       num_points_local_ * sizeof(double));
#endif
    *status = READER_OK;
  }

  double RestartTime() {
    return time_;
  }

  RestartReaderStatus Status() {
    return status_;
  }

private:
  void close_file() {
#ifdef ZERORK_MPI
    MPI_File_close(&restart_file_);
#else
    restart_file_.close();
#endif
  }

#ifdef ZERORK_MPI
  MPI_File restart_file_;
  MPI_Comm comm_;
#else
  std::ifstream restart_file_;
#endif
  int num_fields_;
  int num_points_local_;
  int num_points_file_;
  int rank_;
  int nranks_;
  int global_size_;
  int header_size_;
  double time_;
  std::vector<int> rank_offsets_;
  std::vector<std::string> field_names_;
  RestartReaderStatus status_;
};

} //namespace utilities
} //namespace zerork

#endif
