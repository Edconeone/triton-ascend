// This guard is consumed by the ags-buffer-transaction-audit target.  Its
// database is generated from the finite BufferState universe and declarative
// .ttsem definitions; no copy-forwarding source pattern is supplied to AGS.

// CHECK-LABEL: func.func @private_copy_then_read(
// CHECK-SAME: %[[SOURCE:.*]]: memref<4xi32>
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[SOURCE]] restrict writable
// CHECK: return %[[RESULT]]
func.func @private_copy_then_read(%source: memref<4xi32>) -> tensor<4xi32> {
  %target = memref.alloc() : memref<4xi32>
  memref.copy %source, %target : memref<4xi32> to memref<4xi32>
  %result = bufferization.to_tensor %target restrict writable
    : memref<4xi32> to tensor<4xi32>
  return %result : tensor<4xi32>
}

// An externally owned target is observable after the function returns.  The
// same certified value candidate must therefore be rejected by the generic
// ownership/lifetime gate.
// CHECK-LABEL: func.func @escaping_copy_then_read(
// CHECK: memref.copy
// CHECK: bufferization.to_tensor %{{.*}} restrict writable
func.func @escaping_copy_then_read(%source: memref<4xi32>,
                                    %target: memref<4xi32>) -> tensor<4xi32> {
  memref.copy %source, %target : memref<4xi32> to memref<4xi32>
  %result = bufferization.to_tensor %target restrict writable
    : memref<4xi32> to tensor<4xi32>
  return %result : tensor<4xi32>
}
