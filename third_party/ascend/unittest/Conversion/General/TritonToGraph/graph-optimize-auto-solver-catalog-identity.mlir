// This blind input is intentionally absent from the production-all-ops
// universe. The database is generated from declarative OP semantics only;
// online semantic-normal-form lookup must discover add(x, 0) == x.

// CATALOG-LABEL: tt.func @add_zero(
// CATALOG-NOT: arith.addi
// CATALOG: tt.return %arg0 : i32
tt.func @add_zero(%arg0: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %result = arith.addi %arg0, %zero : i32
  tt.return %result : i32
}
