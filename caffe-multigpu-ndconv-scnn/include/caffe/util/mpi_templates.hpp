#ifdef USE_MPI
#ifndef MPI_TEMPLATES_HPP_
#define MPI_TEMPLATES_HPP_

#include "mpi.h"

template <typename Dtype>
inline int MPIBcast(int count, void* buf, int root = 0,
                    MPI_Comm comm = MPI_COMM_WORLD);
template <>
inline int MPIBcast<float>(int count, void* buf, int root, MPI_Comm comm) {
  return MPI_Bcast(buf, count, MPI_FLOAT, root, comm);
}
template <>
inline int MPIBcast<double>(int count, void* buf, int root, MPI_Comm comm) {
  return MPI_Bcast(buf, count, MPI_DOUBLE, root, comm);
}
template <>
inline int MPIBcast<unsigned int>(int count, void* buf,
                                  int root, MPI_Comm comm) {
  return MPI_Bcast(buf, count, MPI_UNSIGNED, root, comm);
}

template <typename Dtype>
inline int MPIAllreduce(int count, void* sendbuf, void* recvbuf, MPI_Op op,
                        MPI_Comm comm = MPI_COMM_WORLD);
template <>
inline int MPIAllreduce<float>(int count, void* sendbuf, void* recvbuf,
                               MPI_Op op, MPI_Comm comm) {
  return MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, op, comm);
}
template <>
inline int MPIAllreduce<double>(int count, void* sendbuf, void* recvbuf,
                                MPI_Op op, MPI_Comm comm) {
  return MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, comm);
}

template <typename Dtype>
inline int MPIGather(int count, const void* sendbuf, void* recvbuf,
                     int root = 0, MPI_Comm comm = MPI_COMM_WORLD);
template <>
inline int MPIGather<float>(int count, const void* sendbuf, void* recvbuf,
                            int root, MPI_Comm comm) {
  return MPI_Gather(sendbuf, count, MPI_FLOAT, recvbuf, count, MPI_FLOAT,
      root, comm);
}
template <>
inline int MPIGather<double>(int count, const void* sendbuf, void* recvbuf,
                             int root, MPI_Comm comm) {
  return MPI_Gather(sendbuf, count, MPI_DOUBLE, recvbuf, count, MPI_DOUBLE,
      root, comm);
}

template <typename Dtype>
inline int MPIScatter(int count, const void* sendbuf, void* recvbuf,
                      int root = 0, MPI_Comm comm = MPI_COMM_WORLD);
template <>
inline int MPIScatter<float>(int count, const void* sendbuf, void* recvbuf,
                             int root, MPI_Comm comm) {
  return MPI_Scatter(sendbuf, count, MPI_FLOAT, recvbuf, count, MPI_FLOAT,
      root, comm);
}
template <>
inline int MPIScatter<double>(int count, const void* sendbuf, void* recvbuf,
                             int root, MPI_Comm comm) {
  return MPI_Scatter(sendbuf, count, MPI_DOUBLE, recvbuf, count, MPI_DOUBLE,
      root, comm);
}

#endif // MPI_TEMPLATES_HPP_
#endif // USE_MPI