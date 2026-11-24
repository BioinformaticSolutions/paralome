"""Fused stages: the pipeline's intermediates that never become files.

FASTQ QC, alignment and duplicate marking would each write a full copy of every
sample to shared storage for the next one to read back. The trimmed FASTQs and
the aligned-but-not-deduplicated BAM are not results; nothing downstream of
duplicate marking opens either of them, and on a whole-genome run they are the
two largest things the pipeline would move.

This module holds the bash that joins those stages into one task, so the
intermediates exist only as a stream between two processes:

    fastp  raw FASTQ (shared) -> FIFO on this node's scratch
    STAR   FIFO -> unsorted BAM on stdout
    samtools  sort -n | fixmate | sort | markdup -> the one BAM that is kept

Nothing is compressed on the way through, which is the point: a FIFO carries
uncompressed FASTQ, and `--outStd BAM_Unsorted` hands STAR's output to the
sort without a gzip round trip. The work removed is not only I/O -- STAR's own
coordinate sort goes too, because duplicate marking name-sorts and then
position-sorts anyway, so sorting before it would be thrown away.

WHY THE OUTPUT IS THE SAME. The name sort imposes a total order on the reads
regardless of the order STAR emitted them, so the position sort downstream sees
exactly the input it would see from separate stages, and markdup's choice of
which duplicate to keep follows that same order. Compression level and the
location of temporary files do not affect decoded content. The duplicate-marked
BAM is therefore the same as the one three separate stages produce -- a claim
worth checking, by running one sample both ways and comparing the BAMs'
checksums.

WHAT IS GIVEN UP, deliberately:
  * The trimmed FASTQs are not kept. fastp's JSON and HTML reports are
    written to the FASTQ QC stage's own directories, so every QC number
    survives; the reads themselves do not.
  * The aligned BAM is not kept in the paths that deduplicate (DNA, ATAC). The
    RNA path keeps it, because quantification reads it.
  * A task that dies in STAR re-trims on its next attempt: trimming produces
    no file to checkpoint.

HOW A HANG IS AVOIDED. A FIFO whose reader has gone makes its writer block
forever, and a job that blocks burns its whole time limit while looking merely
slow. fastp runs in the background with its PID held; the script waits on it
and checks its status, and an exit trap kills it on every path out, including
SIGTERM from scancel or the time limit.
"""

from __future__ import annotations

import shlex

#: fastp's threads inside a fused task. The cheap end of the stream -- with no
#: compression to do (it writes uncompressed into a FIFO) it trims faster than
#: STAR aligns, so most of the task's CPUs belong to the aligner. Eight rather
#: than four because the first fused run put fastp on the critical path for the
#: opening minutes, before it is far enough ahead to sit in the buffer.
#: (fastp's own cap is 16 regardless of what is asked.)
FUSED_FASTP_THREADS = 8

#: What each of a fused task's two streams gets between fastp and STAR.
#: See paralome.buffer: a kernel pipe is 64 KB, STAR consumes mates in
#: lockstep, and fastp's two output writers run ahead of each other by
#: whatever their internal pack rings allow. Skew beyond the pipe's capacity
#: deadlocks the pair -- which it did, on three of four tasks, at 26 minutes.
FUSED_BUFFER_BYTES = 8 * 1024 ** 3

#: The reader STAR is given in a fused task, in place of `cat`. Shipped with
#: Paralome rather than being mbuffer: a dependency the pipeline cannot run
#: without has no business being something a cluster may not have.
FUSED_READER = "paralome-buffer"


def fused_read_files_command(*, buffer_bytes: int = FUSED_BUFFER_BYTES,
                             reader: str = FUSED_READER) -> str:
    """STAR's --readFilesCommand for a fused task: the buffer, not `cat`.

    `cat` is what the first fused run used, and it is why it deadlocked -- it
    adds no capacity of its own, so the entire channel was two 64 KB pipes.
    """
    return f"{reader} --bytes {int(buffer_bytes)}"


def fused_buffer_overhead(buffer_bytes: int = FUSED_BUFFER_BYTES, *, paired: bool = True) -> int:
    """The RAM a fused task's buffers may hold, which its --mem must cover.

    And the lesson of the first fused run: a fused task's resource
    request has to add up its concurrent pieces instead of inheriting the
    aligner's. That run asked for the CPUs STAR wanted and none of fastp's;
    this is the same mistake's other half, and a task whose buffers sit outside
    its cgroup limit is a task the kernel kills at the worst possible moment.
    """
    return int(buffer_bytes) * (2 if paired else 1)

#: gzip level for trimmed FASTQs that are still WRITTEN as files, in the
#: unfused path. fastp's default is 4. These are read once, by the aligner,
#: and then never again, so the time spent compressing them harder is spent
#: for nobody. Level 1 writes substantially faster for about 10-15% more size.
TRIMMED_COMPRESSION_LEVEL = 1

#: Added wherever Paralome runs fastp. fastp's duplication estimate costs a
#: pass over its hash tables and the memory to hold them, and Paralome marks
#: duplicates properly in its own stage afterwards; the estimate is a number
#: nothing in the pipeline reads.
NO_DUPLICATION_ESTIMATE = "--dont_eval_duplication"


def fastp_speed_args(*, keep_files: bool) -> list[str]:
    """The arguments that make fastp faster without changing which reads it keeps.

    ``keep_files`` is False in a fused task, where the trimmed reads go into a
    FIFO: there is nothing to compress, so -z would be meaningless (fastp picks
    compression from the file name, and a FIFO has no .gz).
    """
    args = [NO_DUPLICATION_ESTIMATE]
    if keep_files:
        args += ["-z", str(int(TRIMMED_COMPRESSION_LEVEL))]
    return args


def fastp_fifo_bash(
    *,
    fastp_exe: str,
    scratch_var: str = "FUSED_SCRATCH",
    sample_var: str = "SAMPLE_ID",
    r1_var: str = "R1",
    r2_var: str = "R2",
    json_dir: str,
    threads: int = FUSED_FASTP_THREADS,
    args_array: str = "TRIM_ARGS",
    paired_array: str = "TRIM_PAIRED_ARGS",
    out_r1_var: str = "TRIM_R1",
    out_r2_var: str = "TRIM_R2",
    pid_var: str = "FASTP_PID",
) -> str:
    """Bash that starts fastp writing into FIFOs and leaves their paths in two variables.

    ``json_dir`` is an already-quoted bash word: the FASTQ QC stage's own report directory (no HTML report
    is written), so a fused run's reports land where an
    unfused run's would and every reader of them is unaffected.

    The FIFOs are named without .gz on purpose -- fastp chooses compression by
    file name, and the whole point is that nothing is compressed here.
    """
    exe = shlex.quote(fastp_exe)
    return "\n".join([
        "# The trimmed reads never become files. They go through two",
        "# FIFOs on this node's scratch, straight into the aligner.",
        f"{out_r1_var}=\"${scratch_var}/${{{sample_var}}}.R1.fastq\"",
        f"{out_r2_var}=\"\"",
        f"mkfifo \"${out_r1_var}\"",
        f"TRIM_JSON={json_dir}/\"${{{sample_var}}}\".json",
        f"mkdir -p {json_dir}",
        f"FASTP_CMD=({exe} -i \"${{{r1_var}}}\" -o \"${out_r1_var}\""
        f" -j \"$TRIM_JSON\" -h /dev/null -w {int(threads)})",
        f"if [[ -n \"${{{r2_var}:-}}\" && \"${{{r2_var}}}\" != \".\" ]]; then",
        f"  {out_r2_var}=\"${scratch_var}/${{{sample_var}}}.R2.fastq\"",
        f"  mkfifo \"${out_r2_var}\"",
        f"  FASTP_CMD+=( -I \"${{{r2_var}}}\" -O \"${out_r2_var}\" )",
        # The ${X[@]+...} form: bash before 4.4 calls an empty array unbound
        # under set -u, as it does everywhere else a Paralome script does this.
        f"  FASTP_CMD+=( ${{{paired_array}[@]+\"${{{paired_array}[@]}}\"}} )",
        "fi",
        f"FASTP_CMD+=( ${{{args_array}[@]+\"${{{args_array}[@]}}\"}} )",
        "echo \"Trimming into a stream: ${FASTP_CMD[*]}\"",
        # If fastp ends before it opens its output (after detecting adapters, say), STAR's reader would sit in
        # open() on the FIFO for ever -- a FIFO opened for reading waits for a writer -- and
        # wait_for_fastp_bash, after STAR, would never be reached. So fastp runs inside
        # paralome_fastp_stream, which, when fastp ends for any reason, holds each FIFO open read-write (on
        # Linux that open does not wait), removes it, and closes it: a reader waiting in open() gets a writer
        # and then end of stream, and one that comes later finds no FIFO and fails. STAR ends on what it was
        # given and the task fails on fastp's own status. A TERM to the task reaches fastp through it.
        "paralome_fastp_stream() {",
        "  \"${FASTP_CMD[@]}\" &",
        "  local child=$! status=0 fifo fd",
        "  trap 'kill -TERM \"$child\" 2>/dev/null || true' TERM",
        "  while :; do",
        "    wait \"$child\" && status=0 || status=$?",
        "    kill -0 \"$child\" 2>/dev/null || break",
        "  done",
        f"  for fifo in \"${out_r1_var}\" ${{{out_r2_var}:+\"${out_r2_var}\"}}; do",
        "    if [[ -p \"$fifo\" ]]; then",
        "      exec {fd}<>\"$fifo\"",
        "      rm -f \"$fifo\"",
        "      exec {fd}>&-",
        "    fi",
        "  done",
        "  return \"$status\"",
        "}",
        "paralome_fastp_stream &",
        f"{pid_var}=$!",
        # A FIFO with no reader blocks its writer for ever, and a blocked job
        # looks like a slow one until the time limit ends it. This runs on
        # every path out of the script, SIGTERM included.
        f"paralome_on_exit kill -TERM \"${pid_var}\"",
    ])


def wait_for_fastp_bash(*, pid_var: str = "FASTP_PID", sample_var: str = "SAMPLE_ID") -> str:
    """Bash that waits for the background fastp and fails the task if it failed.

    Without this the task's status would be STAR's alone, and a sample whose
    trimming died part way would look aligned.

    The error gives fastp's exit status, and the signal when one ended it. One run lost
    two samples to a fastp that stopped part way without a word of its own: a damaged input, a kill and a crash
    could not be told apart from the log.
    """
    return "\n".join([
        "FASTP_STATUS=0",
        f"wait \"${pid_var}\" || FASTP_STATUS=$?",
        "if (( FASTP_STATUS != 0 )); then",
        "  FASTP_WHY=\"exit $FASTP_STATUS\"",
        "  if (( FASTP_STATUS > 128 )); then",
        "    FASTP_WHY=\"$FASTP_WHY: ended by signal $(( FASTP_STATUS - 128 ))"
        " ($(kill -l $(( FASTP_STATUS - 128 )) 2>/dev/null || echo unknown))\"",
        "  fi",
        f"  echo \"ERROR: fastp failed for ${{{sample_var}}} ($FASTP_WHY); the alignment is incomplete\" >&2",
        "  exit 1",
        "fi",
    ])


#: bamUtil's executable (``bam``) and how it is run on the duplicate-marked stream: coordinate-sorted BAM on
#: stdin; the records it skips by default (unmapped, secondary, QC-failed, duplicate, supplementary: 0xF0C)
#: are written unchanged; no usage report sent home.
BAMUTIL_TOOL = "bam"
CLIP_OVERLAP_ARGS = ("--in", "-.bam", "--noPhoneHome")


def markdup_stream_bash(
    *,
    samtools_var: str = "SAMTOOLS",
    threads_var: str = "THREADS",
    scratch_var: str = "DEDUP_SCRATCH",
    remove_flag_var: str = "REMOVE_FLAG",
    output_var: str = "PARTIAL_BAM",
    source: str = "-",
    clip_var: str = "",
) -> str:
    """The four-step duplicate-marking stream, in one place for both callers.

    ``clip_var``: the variable naming bamUtil's ``bam``. The duplicate-marked reads then go
    through ``bam clipOverlap`` (CLIP_OVERLAP_ARGS) before they are written: where the two mates of a pair
    overlap, the overlap is soft-clipped from the mate whose bases there are of lower average quality, so a
    fragment base is counted once by the variant caller.

    ``source`` is the input: a quoted bash word naming a BAM in the duplicate
    marking stage, or ``-`` when it is STAR's output arriving on a pipe.

    WHAT IS DELIBERATELY KEPT, because changing it would change the output:
      * ``sort -n``. ``samtools collate`` is faster and groups mates, but
        it does not order them, so the position sort downstream would receive a
        different order and reads tied at one coordinate could come out in a
        different order -- and markdup's choice of which duplicate to keep
        follows that order.
      * No ``-m``. Changing a sort's memory changes how many temporary chunks
        it makes, and the merge of equal keys follows the chunking.
    """
    marked = (f"  | \"${samtools_var}\" markdup -@ \"${threads_var}\" ${{{remove_flag_var}}}"
              f" -T \"${scratch_var}/markdup\"")
    tail = ([marked + " -u - - \\", f"  | \"${clip_var}\" clipOverlap {' '.join(CLIP_OVERLAP_ARGS)} --out \"${{{output_var}}}\""]
            if clip_var else [marked + f" - \"${{{output_var}}}\""])
    return "\n".join([
        "# shellcheck disable=SC2086",
        f"\"${samtools_var}\" sort -n -@ \"${threads_var}\" -u -T \"${scratch_var}/namesort\" -o - {source} \\",
        f"  | \"${samtools_var}\" fixmate -m -u -@ \"${threads_var}\" - - \\",
        f"  | \"${samtools_var}\" sort -@ \"${threads_var}\" -u -T \"${scratch_var}/possort\" -o - - \\",
        *tail,
    ])


def star_stdout_arguments(*, sorted_output: bool) -> list[str]:
    """How STAR is asked for its alignments: a sorted BAM file, or an unsorted stream.

    In a fused task the sort is removed rather than moved. Duplicate marking
    name-sorts and then position-sorts whatever it is given, so a coordinate
    sort before it was work whose result was immediately thrown away.
    """
    if sorted_output:
        return ["--outSAMtype", "BAM", "SortedByCoordinate"]
    return ["--outSAMtype", "BAM", "Unsorted", "--outStd", "BAM_Unsorted"]
