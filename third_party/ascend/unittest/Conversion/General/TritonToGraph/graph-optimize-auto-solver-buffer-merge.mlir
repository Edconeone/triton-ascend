// RUN: triton-opt %s --graph-optimize="rule-mask=2048 auto-graph-rule-database=%ags_buffer_merge_db" | FileCheck %s
//
// The database is generated from the finite semantic universe, not from this
// topology.  This file is the blind online acceptance test for the static C12
// core: two private buffers become one private buffer with two disjoint writes.

module {
  // CHECK-LABEL: func.func @merge_two_private_buffers
  // CHECK-COUNT-1: memref.alloc
  // CHECK-COUNT-2: memref.copy
  // CHECK-COUNT-1: bufferization.to_tensor
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: tensor.insert_slice
  func.func @merge_two_private_buffers(%source0: memref<8xi32>,
                                       %source1: memref<8xi32>)
      -> tensor<8xi32> {
    %target0 = memref.alloc() : memref<8xi32>
    %target1 = memref.alloc() : memref<8xi32>
    %source0_low = memref.subview %source0[0] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1]>>
    %target0_low = memref.subview %target0[0] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1]>>
    memref.copy %source0_low, %target0_low
      : memref<4xi32, strided<[1]>> to memref<4xi32, strided<[1]>>
    %source1_high = memref.subview %source1[4] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1], offset: 4>>
    %target1_high = memref.subview %target1[4] [4] [1]
      : memref<8xi32> to memref<4xi32, strided<[1], offset: 4>>
    memref.copy %source1_high, %target1_high
      : memref<4xi32, strided<[1], offset: 4>> to memref<4xi32, strided<[1], offset: 4>>
    %tensor0 = bufferization.to_tensor %target0 restrict writable
      : memref<8xi32> to tensor<8xi32>
    %tensor1 = bufferization.to_tensor %target1 restrict writable
      : memref<8xi32> to tensor<8xi32>
    %low = tensor.extract_slice %tensor0[0] [4] [1]
      : tensor<8xi32> to tensor<4xi32>
    %result = tensor.insert_slice %low into %tensor1[0] [4] [1]
      : tensor<4xi32> into tensor<8xi32>
    return %result : tensor<8xi32>
  }
}
