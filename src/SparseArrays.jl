module SparseArrays

using ..Reactant
using ..Reactant: MLIR, Ops, AbstractSparseArray, TracedRArray, TracedRNumber, TracedSparseArray, ConcreteSparseArray
using ..MLIR.Dialects: gpu

export create_csc, create_csr, spmv, spmm

"""
    create_csc(pos, indices, values, rows, cols)

Initialize a sparse matrix in CSC format.
"""
function create_csc(pos, indices, values, rows, cols)
    if pos isa TracedRArray
        location = mlir_stacktrace("create_csc", @__FILE__, @__LINE__)

        # Ensure rows, cols are MLIR values
        rows_val = (rows isa TracedRNumber) ? rows.mlir_data : constant(rows).mlir_data
        cols_val = (cols isa TracedRNumber) ? cols.mlir_data : constant(cols).mlir_data

        # nnz is the number of non-zero elements
        # In traced mode, we might not know nnz exactly, but we can get it from values.shape
        # But typically it's passed as a separate value.
        # If values is a TracedRArray, we might need to compute nnz or pass it.
        # For simplicity, we assume nnz is passed or derived.
        nnz = length(values)
        nnz_val = (nnz isa TracedRNumber) ? nnz.mlir_data : constant(nnz).mlir_data

        # The result type of gpu.create_csc is a sparse matrix descriptor.
        # We use a placeholder type as the actual type is determined by the backend.
        spmat_type = MLIR.IR.Type("!gpu.spmat")

        op = gpu.create_csc(
            MLIR.IR.Value[], rows_val, cols_val, nnz_val,
            pos.mlir_data, indices.mlir_data, values.mlir_data;
            spmat=spmat_type,
            location=location
        )

        res = MLIR.IR.result(op)
        return TracedSparseArray{eltype(values), 2, :CSC}(
            (), res, (rows, cols), nnz
        )
    else
        # Concrete path
        # Assuming rows, cols are Integers and pos, indices, values are ConcreteRArrays
        nnz = length(values)
        return ConcreteSparseArray{eltype(values), 2, :CSC, 1}(
            pos, indices, values, (rows, cols), nnz,
            Sharding.NoShardInfo()
        )
    end
end

"""
    create_csr(pos, indices, values, rows, cols)

Initialize a sparse matrix in CSR format.
"""
function create_csr(pos, indices, values, rows, cols)
    if pos isa TracedRArray
        location = mlir_stacktrace("create_csr", @__FILE__, @__LINE__)

        rows_val = (rows isa TracedRNumber) ? rows.mlir_data : constant(rows).mlir_data
        cols_val = (cols isa TracedRNumber) ? cols.mlir_data : constant(cols).mlir_data

        nnz = length(values)
        nnz_val = (nnz isa TracedRNumber) ? nnz.mlir_data : constant(nnz).mlir_data

        spmat_type = MLIR.IR.Type("!gpu.spmat")

        op = gpu.create_csr(
            MLIR.IR.Value[], rows_val, cols_val, nnz_val,
            pos.mlir_data, indices.mlir_data, values.mlir_data;
            spmat=spmat_type,
            location=location
        )

        res = MLIR.IR.result(op)
        return TracedSparseArray{eltype(values), 2, :CSR}(
            (), res, (rows, cols), nnz
        )
    else
        nnz = length(values)
        return ConcreteSparseArray{eltype(values), 2, :CSR, 1}(
            pos, indices, values, (rows, cols), nnz,
            Sharding.NoShardInfo()
        )
    end
end

"""
    spmv(A, x, y)

Sparse Matrix-Vector multiplication: y = A * x
"""
function spmv(A::AbstractSparseArray{T, N, F}, x, y) where {T, N, F}
    if A isa TracedSparseArray
        location = mlir_stacktrace("spmv", @__FILE__, @__LINE__)

        # 1. Determine buffer size
        # gpu.spmv_buffer_size(asyncDependencies, spmatA, dnX, dnY; bufferSz, asyncToken=nothing, modeA=nothing, computeType, location=Location())

        # We use NON_TRANSPOSE as default modeA
        modeA = MLIR.IR.Attribute("NON_TRANSPOSE")
        compute_type = MLIR.IR.Attribute(" la-default") # Placeholder

        buf_size_op = gpu.spmv_buffer_size(
            MLIR.IR.Value[], A.mlir_data, x.mlir_data, y.mlir_data;
            bufferSz=MLIR.IR.Type(Int64),
            modeA=modeA,
            computeType=compute_type,
            location=location
        )
        buf_size = MLIR.IR.result(buf_size_op)

        # 2. Allocate buffer
        # gpu.alloc(asyncDependencies, dynamicSizes, symbolOperands; memref, asyncToken=nothing, hostShared=nothing, location=Location())
        alloc_op = gpu.alloc(
            MLIR.IR.Value[], [buf_size], MLIR.IR.Value[];
            memref=MLIR.IR.Type("memref<?xi8"), # generic byte buffer
            location=location
        )
        buffer = MLIR.IR.result(alloc_op)

        # 3. Perform SpMV
        # gpu.spmv(asyncDependencies, spmatA, dnX, dnY, buffer; asyncToken=nothing, modeA=nothing, computeType, location=Location())
        spmv_op = gpu.spmv(
            MLIR.IR.Value[], A.mlir_data, x.mlir_data, y.mlir_data, buffer;
            modeA=modeA,
            computeType=compute_type,
            location=location
        )

        # SpMV typically modifies y in place or returns a token.
        # The dialect says it returns a token if async.
        return MLIR.IR.result(spmv_op) # Returning the token or result
    else
        # Concrete path
        error("Concrete SpMV not yet implemented")
    end
end

"""
    spmm(A, B)

Sparse Matrix-Matrix multiplication: C = A * B
"""
function spmm(A::AbstractSparseArray{T, N, F}, B) where {T, N, F}
    if A isa TracedSparseArray
        location = mlir_stacktrace("spmm", @__FILE__, @__LINE__)

        modeA = MLIR.IR.Attribute("NON_TRANSPOSE")
        modeB = MLIR.IR.Attribute("NON_TRANSPOSE")
        compute_type = MLIR.IR.Attribute(" la-default")

        # 1. Buffer size
        # gpu.spmm_buffer_size(asyncDependencies, spmatA, dnmatB, dnmatC; bufferSzs, asyncToken=nothing, modeA=nothing, modeB=nothing, computeType, location=Location())
        # We need dnmatC as well.
        # For now, we assume C is already provided or we create a dummy.
        # This is a simplification.

        # For this implementation, we'll assume B is the dense matrix and we want result C.
        # We need to create C first.
        C_shape = (A.shape[1], size(B)[2])
        C = fill(T(0), C_shape; location)

        buf_size_op = gpu.spmm_buffer_size(
            MLIR.IR.Value[], A.mlir_data, B.mlir_data, C.mlir_data;
            bufferSzs=[MLIR.IR.Type(Int64)],
            modeA=modeA,
            modeB=modeB,
            computeType=compute_type,
            location=location
        )
        buf_size = MLIR.IR.result(buf_size_op)

        # 2. Allocate buffer
        alloc_op = gpu.alloc(
            MLIR.IR.Value[], [buf_size], MLIR.IR.Value[];
            memref=MLIR.IR.Type("memref<?xi8"),
            location=location
        )
        buffer = MLIR.IR.result(alloc_op)

        # 3. Perform SpMM
        # gpu.spmm(asyncDependencies, spmatA, dnmatB, dnmatC, buffers; asyncToken=nothing, modeA=nothing, modeB=nothing, computeType, location=Location())
        spmm_op = gpu.spmm(
            MLIR.IR.Value[], A.mlir_data, B.mlir_data, C.mlir_data, [buffer];
            modeA=modeA,
            modeB=modeB,
            computeType=compute_type,
            location=location
        )

        return MLIR.IR.result(spmm_op)
    else
        error("Concrete SpMM not yet implemented")
    end
end

end
