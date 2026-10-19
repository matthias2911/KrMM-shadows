using RandomMeas
using ITensors
using NPZ
using ArgParse
using Base.Threads

ITensors.disable_warn_order()

# -----------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------

const Pauli = Symbol # :X, :Y, :Z

const pauli_map = Dict(
    0 => :Z,
    1 => :X,
    2 => :Y
)

function pauli_unitary(pauli::Symbol, s::Index)
    sp = prime(s)
    U = ITensor(sp, s)

    if pauli == :Z
        U[s=>1, sp=>1] = 1
        U[s=>2, sp=>2] = 1
    elseif pauli == :X
        U[s=>1, sp=>1] = 1 / sqrt(2)
        U[s=>2, sp=>1] = 1 / sqrt(2)
        U[s=>1, sp=>2] = 1 / sqrt(2)
        U[s=>2, sp=>2] = -1 / sqrt(2)
    elseif pauli == :Y
        U[s=>1, sp=>1] = 1 / sqrt(2) + 0im
        U[s=>2, sp=>1] = 0 + im / sqrt(2)
        U[s=>1, sp=>2] = 1 / sqrt(2) + 0im
        U[s=>2, sp=>2] = 0 - im / sqrt(2)
    else
        error("Unknown Pauli $pauli")
    end

    return U
end

# -----------------------------------------------------------------------
# NPZ import: NU unitaries, NM outcomes each.
# -----------------------------------------------------------------------
function import_MeasurementGroupNPZ(filepath::String;
                                    site_indices=nothing,
                                    order::Union{Int,Nothing}=nothing)
    println("Loading npz file: $filepath")
    data = npzread(filepath)

    N          = Int(data["num_qubits"][1])
    NU_file    = Int(data["num_unitaries"][1])
    NM         = Int(data["num_meas"][1])        # outcomes per unitary
    order_file = Int(data["max_order"][1])

    order_used = order === nothing ? order_file : order
    @assert order_used <= order_file "Requested order $order_used > order in file ($order_file)"

    # order_used blocks of all 3^N settings
    stratified = 3^N * order_used <= NU_file
    if stratified
        NU = 3^N * order_used
    else
        batch_size = div(NU_file, order_used)
        @assert batch_size >= 1 "File has only $NU_file unitaries for $order_used batches"
        NU = batch_size * order_used          # equal batches, drops at most order_used-1 unitaries
        println("File has $NU_file < 3^N * order = $(3^N * order_used) unitaries: random-settings mode, " *
                "$order_used batches of $batch_size unitaries ($(NU_file - NU) dropped)")
    end

    # C-order flattening
    inds   = reshape(data["inds"],   N, NU_file)       # inds[q, u]      in {0,1,2}
    clicks = reshape(data["clicks"], N, NM, NU_file)   # clicks[q, m, u] in {0,1}

    # each batch should contain every one of the 3^N settings exactly once
    if stratified
        for b in 1:order_used
            cols = ((b - 1) * 3^N + 1):(b * 3^N)
            if length(unique(eachcol(inds[:, cols]))) != 3^N
                @warn "Batch $b does not contain all 3^N distinct settings"
            end
        end
    end

    site_indices === nothing && (site_indices = siteinds("Qubit", N))

    unitary_cache = Dict{Vector{Int},Vector{ITensor}}()

    measurements = Vector{MeasurementData{LocalUnitaryMeasurementSetting}}(undef, NU)
    for u in 1:NU
        key = Int.(inds[:, u])
        local_unitaries = get!(unitary_cache, key) do
            [pauli_unitary(pauli_map[key[j]], site_indices[j]) for j in 1:N]
        end
        ms = LocalUnitaryMeasurementSetting(N, local_unitaries, site_indices)

        m_results = Int.(permutedims(clicks[:, :, u])) .+ 1   # (NM, N), {0,1} -> {1,2}
        measurements[u] = MeasurementData(N, NM, m_results, ms)
    end

    println("Loaded $NU MeasurementData objects ($NM outcomes each), $order_used batches of $(div(NU, order_used)), $N qubits")
    return MeasurementGroup(measurements), N, NU, NM, order_used, data
end

# From: https://github.com/bvermersch/RandomMeas.jl/blob/main/src/DenseShadows.jl
# Changes: parallelized over batches
function get_dense_shadows(
    measurement_group::MeasurementGroup{LocalUnitaryMeasurementSetting};
    G::Vector{Float64} = fill(1.0, measurement_group.N),
    number_of_ru_batches::Int = measurement_group.NU
)
    NU, N = measurement_group.NU, measurement_group.N
    data = measurement_group.measurements
    ξ = data[1].measurement_setting.site_indices

    @assert length(G) == N "Length of G must match the number of qubits/sites."

    batch_size = div(NU, number_of_ru_batches)
    ru_batches = [((b - 1) * batch_size + 1):(b == number_of_ru_batches ? NU : b * batch_size) for b in 1:number_of_ru_batches]

    shadows = Vector{DenseShadow}(undef, number_of_ru_batches)
    println("Using threads: $(Threads.nthreads()) to compute dense shadows in $number_of_ru_batches batches.")
    println("NU = $NU.")

    Threads.@threads for batch_id in eachindex(ru_batches)
        ru_batch = ru_batches[batch_id]

        batch_shadow = ITensor(vcat(ξ, prime(ξ)))
        for r in ru_batch
            batch_shadow += DenseShadow(data[r]; G=G).shadow_data
        end

        shadows[batch_id] = DenseShadow(
            batch_shadow / length(ru_batch),
            N,
            ξ
        )
    end

    return shadows
end

# -----------------------------------------------------------------------
# All PT moments 2...order from order batches
# -----------------------------------------------------------------------
function get_trace_moments_merged(dense_shadows::Vector{<:DenseShadow},
                                  pt_sites::Vector{Int},
                                  N::Int,
                                  ξ)
    order = length(dense_shadows)
    @assert order >= 2 "Need at least 2 batches"

    raw    = collect(dense_shadows)  # non-transposed groups
    pt     = collect(partial_transpose(dense_shadows, pt_sites)) # transposed groups
    counts = ones(Int, order)                           # original batches per group

    moments = zeros(Float64, order)                         
    moments[order] = real(get_trace_product(pt...))

    for k in (order - 1):-1:2
        # two smallest groups
        perm = sort(1:length(counts); by = i -> (counts[i], -i))
        i, j = minmax(perm[1], perm[2])

        c = counts[i] + counts[j]
        merged = (counts[i] * raw[i].shadow_data + counts[j] * raw[j].shadow_data) / c

        raw[i]    = DenseShadow(merged, N, ξ)
        pt[i]     = partial_transpose([raw[i]], pt_sites)[1]
        counts[i] = c

        deleteat!(raw, j)
        deleteat!(pt, j)
        deleteat!(counts, j)

        moments[k] = real(get_trace_product(pt...))
    end

    return moments[2:order]
end

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
function main(parsed_args)
    in_path  = parsed_args["input"]
    out_path = parsed_args["output"]
    m_arg    = parsed_args["m"]    

    if isfile(out_path)
        println("Results already exist at $out_path, skipping computation.")
        return
    end

    mg, n_qubits, NU, NM, order, raw_data =
        import_MeasurementGroupNPZ(in_path; order = m_arg > 0 ? m_arg : nothing)
    @assert order >= 2 "Need order >= 2 to estimate moments 2..order"

    moments = Float64[]
    elapsed_time = @elapsed begin

        println("Start building ", order, " batches")        
        start = time()

        # order batches with 3^N settings each
        dense_shadows = get_dense_shadows(mg; number_of_ru_batches=order)

        elapsed_batching = time() - start
        println("Runtime building batches: ", elapsed_batching, " seconds")

        # equal split PT
        NA = div(n_qubits, 2)
        pt_sites = collect(NA+1:n_qubits)
        ξ = mg.measurements[1].measurement_setting.site_indices

        moments = get_trace_moments_merged(dense_shadows, pt_sites, n_qubits, ξ)
    end

    orders = collect(2:order)
    exact_moments = Float64.(raw_data["moments_pt"][1:order])
    pushfirst!(moments, 0.0)
    for (idx, k) in enumerate(orders)
        println("PT moment k=$k: est = $(moments[idx+1]), exact = $(exact_moments[idx+1])")
    end

    println("Total elapsed time: $elapsed_time seconds.")

    out_dict = Dict{String,Any}(
        "num_qubits"      => Int32[n_qubits],
        "num_meas"        => Int32[NM],
        "num_unitaries"   => Int32[NU],
        "order"           => Int32[order],
        "num_threads"     => 1,
        "vector_width"    => 1,
        "PT"              => 1,
        "werner"          => raw_data["werner"],
        "werner_par"      => raw_data["werner_par"],
        "var_offset"      => 2500,
        "rmMB"            => 1,
        "rsptMB"          => 1,
        "rfMF"            => 1,
        "rmtMB"           => 1,
        "num_runs"        => 1,
        "num_data_points" => 0,
        "runtime"         => Float64[elapsed_time],
        "est_moments"     => moments,                       
        "est_std_errs"    => zeros(Float64, order - 1),      
        "transposes"      => raw_data["transposes"],
        "moments"         => exact_moments,                  
        "num_batches"     => Int32.(orders),                 
    )

    npzwrite(out_path, out_dict)
    println("Results written to $out_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    s = ArgParseSettings()
    @add_arg_table s begin
        "--input"
            help = "Path to input .npz file containing measurement data"
            arg_type = String
            required = true
        "--output"
            help = "Path to output .npz file to write results to"
            arg_type = String
            required = true
        "--m"
            help = "Max moment order / number of batches (0 = use max_order from file)"
            arg_type = Int
            default = 0
    end

    parsed_args = parse_args(s)
    main(parsed_args)
end