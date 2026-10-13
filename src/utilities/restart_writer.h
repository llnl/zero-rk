#ifndef RESTART_WRITER_H
#define RESTART_WRITER_H


#ifdef ZERORK_MPI
#include <mpi.h>
#else
#include <fstream>
#endif
#include <assert.h>
#include <string>
#include <vector>

namespace zerork {
namespace utilities {

class RestartWriter {

public:
  RestartWriter(std::string filename, int num_points, int num_fields,
                double time, std::vector<std::string> field_names
#ifdef ZERORK_MPI
                ,
                MPI_Comm comm
#endif
                )
      : num_points_local_(num_points), num_fields_(num_fields),
        num_fields_written_(0), time_(time), rank_(0), nranks_(1)
#ifdef ZERORK_MPI
        ,
        comm_(comm)
#endif
  {
    assert(num_fields == field_names.size());

#ifdef ZERORK_MPI
    MPI_File_open(comm_, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR,
                  MPI_INFO_NULL, &output_file_);

    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nranks_);

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

    if (rank_ == 0) {
      MPI_File_write(output_file_, &global_size_, 1, MPI_INT,
                     MPI_STATUS_IGNORE);
      MPI_File_write(output_file_, &num_fields_, 1, MPI_INT, MPI_STATUS_IGNORE);
      MPI_File_write(output_file_, &time_, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
      char name_buf[64];
      for (int i = 0; i < num_fields_; ++i) {
        strncpy(name_buf, field_names[i].c_str(), 64);
        MPI_File_write(output_file_, name_buf, 64, MPI_CHAR, MPI_STATUS_IGNORE);
      }
      MPI_File_seek(output_file_, 0, MPI_SEEK_SET);
    }
    header_size_ =
        2 * sizeof(int) + sizeof(double) + num_fields_ * sizeof(char) * 64;
#else
    output_file_.open(filename.c_str(), ios::binary | ios::out);

    output_file_.write((char *)&num_points_local_, sizeof(int));
    output_file_.write((char *)&num_fields_, sizeof(int));
    output_file_.write((char *)&time_, sizeof(double));
    for (int j = 0; j < field_names.size(); j++) {
      char buf[64];
      strncpy(buf, field_names[j].c_str(), 64);
      output_file_.write((char *)&buf, 64 * sizeof(char));
    }
#endif
  }

  ~RestartWriter() {
#ifdef ZERORK_MPI
    MPI_File_close(&output_file_);
#else
    output_file_.close();
#endif
  }

  void WriteField(const std::vector<double> &field) {
    assert(field.size() == num_points_local_);

#ifdef ZERORK_MPI
    int disp = header_size_;
    disp += num_fields_written_ * global_size_ * sizeof(double);
    disp += rank_offsets_[rank_] * sizeof(double);
    MPI_File_write_at(output_file_, disp, field.data(), num_points_local_,
                      MPI_DOUBLE, MPI_STATUS_IGNORE);
#else
    output_file_.write((char *)field.data(),
                       num_points_local_ * sizeof(double));
#endif
    num_fields_written_ += 1;
  }

private:
#ifdef ZERORK_MPI
  MPI_File output_file_;
  MPI_Comm comm_;
#else
  ofstream output_file_;
#endif
  int num_fields_;
  int num_points_local_;
  int rank_;
  int nranks_;
  int global_size_;
  int header_size_;
  int num_fields_written_;
  double time_;
  std::vector<int> rank_offsets_;
};

} //namespace utilities
} //namespace zerork

#endif
