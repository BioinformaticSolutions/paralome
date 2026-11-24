# Installing Paralome

**For the person who administers the machine.** If you are here to run an
analysis, you want `MANUAL.md` instead — this document is about setting up the
place the analyses will run.

Paralome is a Python package. Installing it is `pip install`. Everything else in
this document exists because of one thing: **a reference genome is expensive to
build, large on disk, and identical for everybody who uses it.** A site where
four people each keep their own copy of GRCh38 is wasting a quarter of a
terabyte and, worse, is four separate answers to the question "which annotation
release were those results against?"

So Paralome has a way for one person to build a genome and everyone else to use
it without copying it, retyping its paths, or being given access to each other's
files. Setting that up is what the rest of this is.

---

## 1. Decide which kind of machine this is

Paralome behaves differently in three situations, and it will infer which one
you are in. Read the one that matches.

| | Who uses it | What is shared |
|---|---|---|
| **Workstation** | One person, or a few who trust each other completely | Nothing needs to be; genomes can live in a home directory |
| **Shared server** | Several people with private home directories | A genome repository and a registry, both outside anyone's home |
| **Cluster** | Several people, several machines, a shared filesystem | The same, plus identity consistency between machines |

The deployment is inferred from the machine itself each time it is needed —
SLURM present means cluster, three or more human accounts means shared — and
is not written anywhere. You can see what it decided, and from which signal,
in the Scheduler section of:

```
paralome setup doctor
```

You do not have to configure it. A site that disagrees with the inference sets
`defaults.deployment` to `workstation`, `shared` or `cluster` in the config,
and that recorded value wins. It changes only what Paralome *advises*, never
what it permits.

---

## 2. Install the package

Paralome needs Python 3.10 or newer (the typer it uses requires it).

**System-wide**, which is what you want on a shared machine — one copy, one
version, everyone gets the same thing:

```
sudo python3 -m pip install paralome
which paralome
paralome --version
```

**`pip install --user` is refused.** Paralome will not run from a copy
installed into one account's home directory; it stops with PLM-0408. An install
belongs to the machine, so that everybody on it shares one version and one set
of reference genomes — a private copy can neither publish to the shared
registry nor see what is in it, so it silently rebuilds references that already
exist. If you do not have the access to install for the machine, the install is
not yours to make.

A virtualenv is not affected. The test is for user-site specifically, not for
whether root was needed, so development environments work normally.

Confirm every account resolves the same binary:

```
sudo -iu <some-account> which -a paralome
```

### The two dependencies worth checking

Paralome pins these deliberately, and a machine that resolves outside the pin
fails in ways that do not look like dependency problems:

```
python3 -m pip list | grep -iE 'typer|rich'
```

| | pinned | why the ceiling exists |
|---|---|---|
| `typer` | `>=0.27,<0.28` | typer is pre-1.0 and its minor releases change behaviour; a fresh install gets the one that was tested |
| `rich` | `>=13.8,<16` | rich draws the grouped help; without a workable one the grouping silently vanishes |

Out of the pin? Put both back inside it:

```
sudo python3 -m pip install 'typer>=0.27,<0.28' 'rich>=13.8,<16'
```

**click is not a dependency**: typer carries its own copy of it. A click
installed by something else is not used by Paralome and need not be removed.

The rich ceiling fails quietly, which is worse in its own way: the help still
prints and still lists every option, just as one undifferentiated block with
the per-stage headings gone.

**Every machine that runs Paralome should report the same two versions.**
If they differ, one wheel is running against two stacks.

### The external tools

Paralome orchestrates other people's software. It does not bundle it.

| Tool | Needed for |
|---|---|
| `fastp` | Read trimming and QC |
| `STAR` | RNA and ATAC alignment |
| `samtools`, `bcftools`, `bgzip`, `tabix` | Everywhere |
| `freebayes` | Variant calling |
| `java` | snpEff |
| `sbatch`, `squeue`, `sacct` | Only if you use SLURM |

Check them with:

```
paralome setup doctor
```

Anything missing is named. Paralome will not report itself ready while a core
tool is absent.

The single-cell pipelines (snATAC and scRNA) also need `Rscript` with the
`Seurat`, `Signac` and `Matrix` packages, to read the Seurat object. `setup
doctor` does not check for it and `setup` does not record it: it is run as
`Rscript` from the job's PATH, or as the installation `--rscript` names.

**Paralome records where each tool is, and then uses what it recorded.** The
first `paralome setup` on an account writes down the absolute path of every
tool in the table above it finds (not `Rscript`), and every run from then on
invokes those paths rather than
searching PATH again. If a recorded path stops resolving — a tool upgraded into
a different directory, say — the next run **stops** with PLM-0410 instead of
quietly running a different binary. Re-running `paralome setup` re-records them.

That makes PATH a setup-time concern rather than a run-time one, but it still
has to be right *at setup time*, and it has to be right for everybody:

**Put the tools on PATH for everyone, not in one person's dotfiles.** If your
tools live somewhere unusual, a drop-in is the right place:

```
cat > /etc/profile.d/paralome_tools.sh <<'EOF'
case ":$PATH:" in
    *":/opt/bio/bin:"*) ;;
    *) PATH="$PATH:/opt/bio/bin" ;;
esac
export PATH
EOF
chmod 644 /etc/profile.d/paralome_tools.sh
```

Two details worth the trouble. **Append rather than prepend** unless you mean to
override the system copies — a directory of hand-built tools placed ahead of
`/usr/bin` gives some accounts an older `samtools` than others, and because each
account records whatever its own PATH found at setup time, that difference gets
frozen into their configs. `paralome setup doctor` reports it when a recorded
path and PATH disagree, but the cheaper fix is not to create the disagreement.
And **check the file before you trust it**: a syntax error in
`/etc/profile.d/*.sh` breaks login for every account including root, so run
`bash -n` on it and test a fresh login from a second terminal while your first
one is still open.

**Every account runs `paralome setup` once**, after the tools are in place and
after they have the group. It is not optional: an account that has never run it
has recorded nothing, and an account that ran it before you moved the tools has
recorded the wrong thing.

---

## 3. A workstation

Nothing to set up. The user runs:

```
paralome setup
paralome setup genome add /path/to/a/genome/folder
```

Genomes can live anywhere readable. Skip to section 8.

---

## 4. A shared server

This is the case the sharing machinery exists for: several people, private home
directories, and no reason for any of them to read each other's files.

### 4.1 What you are building

One directory, with two things under it:

```
/storage/paralome/genomes     the genome FILES -- fasta, gtf, indexes
/storage/paralome/registry    the REGISTRY -- who has published what
```

Put the root anywhere with room for genome indexes — tens to hundreds of
gigabytes — that is **not inside anyone's home directory** and that every
account can traverse. On a cluster it must be on the shared filesystem, visible
from the compute nodes as well as the head node.

The first is bulk data. The second is a small directory of JSON files, one per
account, describing which genomes exist and where their files are. A user reads
the registry to discover genomes and adopts the entries they want; **nothing
ever reads anybody's home directory.**

The two are siblings rather than nested, deliberately. A registry inside the
genome folder is one `chown -R` away from losing its mode, and people run
recursive commands on genome repositories.

Where you put the root is recorded on the machine, in
`<install prefix>/share/paralome/site.json` — usually
`/usr/share/paralome/site.json`. Every account reads that file, so a person who
has never seen this machine before finds the genomes without being told where
they are. That is the whole point: an account created next month has a fresh
config and would otherwise have no idea, and the natural thing to do with no
idea is build a private copy.

### 4.2 Home directories stay private

Paralome can discover genomes by reading other accounts' configuration files
directly out of their home directories. **On a shared machine, do not enable
this by opening homes.** It works, and it is the wrong thing to build on: no
managed multi-user system will permit it, and a site that opens its homes to
make a tool work has traded a real security property for a convenience that the
registry already provides.

Homes should be `0700`. Check yours:

```
getent passwd | awk -F: '$3>=1000 && $3<65534 {print $6}' | xargs -r ls -ld
```

If some are world-readable, close them — but only ever by *removing* access,
never by widening it, and never touching ownership or contents:

```
chmod o-rwx /path/to/that/home
```

### 4.3 The directory that holds the homes

Before anything else, check the parent of your home directories:

```
ls -ld /path/that/contains/the/homes
```

**If it is world-writable and not sticky, fix that first**, and it has nothing
to do with Paralome. In Unix, permission to delete or rename an entry comes from
write+execute on the *parent directory*, not from the entry itself. A `0777`
directory holding home directories means any account on the machine can do:

```
mv /path/to/homes/SomeoneElse /tmp/gone
```

with no root, no ownership check and no warning. The sticky bit is what
normally restricts that to each entry's owner — `/tmp` is `1777` for exactly
this reason.

If the homes are created by root, the tighter fix is better:

```
chmod 0755 /path/that/contains/the/homes
chown root:root /path/that/contains/the/homes
```

If something unprivileged legitimately creates directories there, use `chmod +t`
instead and leave it writable.

This matters to Paralome specifically because a user's configuration lives in
their home. A writable path to those files means anyone can forge one — a fake
withdrawal to block a colleague's work, or a genome entry pointing at files of
their choosing. The trust model assumes only the owner can write their own
config; the filesystem is what enforces it.

### 4.4 Build it

One command, as root:

```
paralome setup site --root /storage/paralome
```

**That is a dry run.** It prints every directory it would create, the mode and
ownership it would set, what each mode bit is for, and whether the group needs
creating — and writes nothing. Read it, then repeat with `--apply`:

```
paralome setup site --root /storage/paralome --apply
```

which creates the group if it is absent, creates both directories, sets their
ownership and modes, and records the root in `site.json`.

Doing this by hand is eight commands, two of which are numeric modes encoding
three separate decisions each and looking like typos. Getting one wrong
produces a machine that works until the second person uses it — so the command
does it, and the dry run shows you exactly what it did.

**Afterwards, bare `paralome setup site` reports what this machine decided**
without changing anything: where `site.json` is, the recorded root, and the two
directories derived from it. It also says how to change it, which is the
question that otherwise has no discoverable answer.

**Changing it later** is the same command with a different `--root` — still a
dry run by default. It will tell you what the move does *not* do, which is the
part worth reading twice: nothing is moved. Genomes already registered keep the
absolute paths recorded against them, registry files written to the old
location stay there and are simply no longer read, and moving the contents is
yours to do. Paralome changes where it looks, not where anything is.

**The group is not recorded in `site.json`,** and that is deliberate rather than
an omission — it is carried by the `genomes` directory itself, set at creation
and inherited by everything added since, so a second copy of the answer could
only ever disagree with it. `paralome setup doctor` checks both directories'
modes, and that the default registry belongs to group `root`; it does not
check either directory's owner or the repository's group.

**`site.json` is checked key by key, like every configuration file.** As of
A key in it that nothing reads is refused with **PLM-0411**, and a
file that is not valid JSON with **PLM-0412**. Because every account reads
this file, one mistyped key in it stops every run by every account on the
machine until an administrator corrects it; the message names the file, the
key and the valid ones, and `paralome setup doctor` reports it. If you edit
the file by hand, run `paralome setup site` afterwards to confirm it still
loads. `setup site --apply` keeps the keys already recorded.

The same check applies to each account's `~/.paralome/config.json`, and there
is no default genome any more: a `defaults.genome` in an account's
configuration is accepted and not used, and every analysis needs `--genome`.

What it sets, and why:

| | Mode | Owner | |
|---|---|---|---|
| `<root>/genomes` | `3775` | `root:paralome` | setgid + sticky + group write |
| `<root>/registry` | `1777` | `root:root` | the `/tmp` arrangement |
| `site.json` | `0644` | `root:root` | every account must be able to read it |

`3775` on the repository is three decisions:

- **group write** — any member of the group may add a genome
- **setgid** — what they add inherits the group, so the next person can read it
  without anyone remembering to `chgrp`
- **sticky** — but only the owner may delete or rename their own genome, which
  is the same hole described in 4.3, deliberately not reopened here

`1777` on the registry is the same reasoning: everyone may publish their own
file, nobody may remove another's. It looks alarming and is not — nobody can
modify another account's file, the files are small JSON documents describing
paths rather than containing data, and a malformed one is skipped rather than
taking everyone else's genomes down.

Use `--group` if `paralome` is not the group name you want.

**Nothing is made recursive.** Existing genome folders keep their current group
and mode; only new contributions inherit.

#### Group write on new content

Setgid gives new content the right *group*. It does not give it group *write* —
that comes from each contributor's umask, which Paralome cannot set. On a
filesystem with POSIX ACLs, `setup site --apply` sets a default ACL that makes
new content group-readable whatever the umask. On NFS it cannot: `setfacl`
returns `EOPNOTSUPP`, and the command says so and prints the alternative, which
is to pin the umask machine-wide:

```
echo 'umask 022' > /etc/profile.d/paralome_umask.sh
chmod 644 /etc/profile.d/paralome_umask.sh
```

Check it took, from a *fresh* login — an open session keeps the umask it
started with:

```
sudo -iu <account> sh -c umask     # expect 0022
```

### 4.5 The group is membership, not ownership

`setup site` creates the group. It does not put anyone in it, because who
belongs is not something a program should decide:

```
usermod -aG paralome alice
usermod -aG paralome bob
```

**Setting the group on someone's files does not make them a member of that
group.** Access comes from the process's group set. `chown -R alice:paralome
~alice` grants alice nothing and mislabels her home. Use `usermod -aG`.

Group changes take effect at **next login**. A session that is already open
keeps the groups it started with, so tell people to log out and back in, and
check with `id -nG alice` rather than assuming.

### 4.6 Verify it

As two different unprivileged accounts:

```
# alice
paralome setup
paralome setup genome add /storage/paralome/genomes/Homo_sapiens.GRCh38.e113

# bob
paralome setup
paralome setup genome list      # should show alice's genome
paralome setup genome sync      # should offer to adopt it
```

And confirm the boundaries hold:

```
# bob, who should not be able to read alice's home
sudo -iu bob ls ~alice          # expect: Permission denied

# bob, who should be able to contribute
sudo -iu bob touch /storage/paralome/genomes/probe && \
    ls -l /storage/paralome/genomes/probe   # expect: group paralome, via setgid
rm /storage/paralome/genomes/probe

# bob, who must NOT be able to remove alice's genome
sudo -iu bob rm -rf /storage/paralome/genomes/Homo_sapiens.GRCh38.e113
                                # expect: Operation not permitted, via sticky
```

That last one is the check worth running. Group write without the sticky bit
gives every member the ability to delete every other member's genome, and the
difference is invisible until the day somebody does.

---

## 5. A cluster

Everything in section 4 applies. Three things are added by having more than one
machine.

### 5.1 Identity must match across machines

**On a shared filesystem, identity is numeric.** The filesystem stores UIDs and
GIDs; each machine resolves those numbers to names using its own
`/etc/passwd` and `/etc/group`. The names are local decoration.

So if you create accounts independently on each node with `useradd`, the two
allocation counters drift the moment anything is created on one and not the
other — and then the same person is two different identities depending on where
they logged in, or worse, two *different* people share one UID and the
filesystem considers them the same person.

Either run a central identity service (LDAP, FreeIPA with SSSD — what a
university HPC does), or pin the numbers explicitly:

```
groupadd -g 1501 alice
useradd -u 1501 -g 1501 -G paralome -m -d /storage/home/alice -s /bin/bash alice
```

and run the identical command, with the identical numbers, on every node.

Check agreement by comparing both machines:

```
getent passwd | awk -F: '$3>=1000 && $3<65534 {printf "%s %s\n", $3, $1}' | sort -n
getent group  | awk -F: '$3>=1000 && $3<65534 {printf "%s %s\n", $3, $1}' | sort -n
getent group paralome
```

The same number mapping to different names is a problem. The same name mapping
to different numbers is a problem. `paralome` missing on one node means
group-based access works on one machine and silently does not on the other.

Renumbering an existing account is not a `usermod`: changing a UID orphans every
file that account owns, so the real operation is stop the account being used,
`usermod -u`, `find /shared -uid <old> -exec chown <new> {} +`, and the same for
the group. Get it right when you create the account instead.

### 5.2 Allocate new numbers above the highest in use

Not into the first gap. A gap is usually a deleted account, and shared storage
remembers it — files under that UID outlive the account, so reissuing the number
hands a new person their predecessor's data. Numbers are not scarce; recovered
data is.

### 5.3 Install everywhere, and check PATH everywhere

Install the same version on every node. A login node one release ahead of
the compute nodes produces failures that look like data problems.

SLURM exports the submitting environment by default, so a job inherits the login
node's PATH — which means a PATH entry pointing at a directory that exists only
on the login node resolves to nothing on the compute node.

Paralome narrows this considerably: submission scripts resolve each recorded
tool on the **login node at submit time** and write the absolute path into the
script (`Rscript` excepted -- it is found on the job's PATH), so a
compute node's own PATH never decides which binary runs. But the path still has
to exist over there. A tool installed only on the head node is now a job that
fails immediately with a clear message rather than one that finds a different
version and finishes, which is the better failure but still a failure.

Verify from a node, not from the submit host:

```
ssh <compute-node> 'paralome --version; command -v STAR samtools'
```

Every node should report the version you just installed and the same tool
paths the login node records.

The genome root must be on the shared filesystem too, for the same reason —
`site.json` names one path and every node has to be able to reach it.

---

## 6. Registering the first genomes

A genome is a folder. Put it in the repository — `genome add` refuses a folder
anywhere else once the machine has a recorded root, because a genome registered
outside it can only be used by the account that registered it, which is the
duplication all of this exists to end.

Paralome scans the folder and works out what is inside:

```
paralome setup genome add /storage/paralome/genomes/Human_GRCh38_113
```

A bare name works too, resolved against the recorded repository:

```
paralome setup genome add Human_GRCh38_113
```

It will show what it found and ask you to confirm five identity fields:

```
Species             Homo sapiens
Assembly            GRCh38
Annotation source   ensembl
Annotation release  113
Modified            no
```

Answer them properly. The assembly and the annotation are **versioned
independently** — GRCh38 has had dozens of Ensembl releases — so a genome
recorded with only its assembly cannot be compared with another for
supersession, and Paralome will tell you it takes no part in it rather than
staying quiet.

`unknown` is a real answer if you do not know. It is recorded as unknown, which
is honest, rather than guessed.

### The name is derived, not chosen

Those fields matter more than they look, because **the genome's name comes from
them**:

```
<Species>.<Assembly>.<source-initial><release>

Homo_sapiens.GRCh38.e81
Mus_musculus.GRCm39.e112
Ovis_aries_rambouillet.ARS-UI_Ramb_v2_0.e113
```

You do not pick it and there is no `--name`. The genome is named for what it
is, and its folder is named for the genome.

**The folder carries the genome's name**. When the folder is called
something else, `genome add` says so, renames the folder to the derived name,
registers it from there and reports the rename. When it cannot -- a folder of
that name is already there, or the account may not rename it -- nothing is
registered, and it prints the name to give the folder and the two commands to
run:

```
Cannot rename the folder: /storage/paralome/genomes/Homo_sapiens.GRCh38.e113 already exists.
Nothing was registered. A genome's folder must carry the genome's name,
Homo_sapiens.GRCh38.e113. Rename the folder to that, then register it:

  mv /storage/paralome/genomes/human_38 /storage/paralome/genomes/Homo_sapiens.GRCh38.e113
  paralome setup genome add /storage/paralome/genomes/Homo_sapiens.GRCh38.e113
```

So the name a listing shows is the folder on disk, and the folder on disk is the
name a run takes. Moving the folder loses nothing: the FASTA and GTF are found
by the file names the STAR index recorded, not by their old paths.

The reason is duplication again. When names are free text, two people
registering one reference produce `Human_GRCh38.e113` and `Human38` — the same
duplicate, with nothing in common for any check to catch. Derived names collide
by construction, so the second person is told the genome already exists and
offered the existing registration instead of publishing a rival claim to it.

It also means the identity you confirm at registration is load-bearing. A
genome whose assembly you typed carelessly gets a name that says so, and every
listing repeats it.

The names are long. Nobody types them — `paralome setup genome list` prints
them in full and people copy from there.

**Flag anything you have modified:**

```
paralome setup genome add Human_GRCh38_GFP \
    --modified --modified-note "GFP and mCherry added to assembly and GTF"
```

Fluorescent reporters added to an assembly and GTF make a genome that is correct
for your experiment and wrong for everybody else's. The flag travels with the
entry and appears in every listing.

### Motif databases are shared, once

JASPAR CORE vertebrates is the same file for human, macaque, mouse and sheep —
the collection is taxonomic, not per-assembly. Keep one copy **inside the
repository, beside the genome folders**, and every genome registered afterwards
will find it:

```
/storage/paralome/genomes/motifs/jaspar/<release>/JASPAR<release>_..._meme.txt
```

The location is not a suggestion. The scan looks inside the genome folder
first, then exactly one level up — the repository root — and no further,
because a rule that kept climbing would eventually reach a home directory and
start borrowing motif files belonging to unrelated work. Put the motifs
anywhere else and every genome registers without one.

Move the motifs **before** anyone registers a genome. A genome registered while
the motif file is missing records no motif path, and re-registering is the only
way to give it one.

A genome that needs its own — a plant, an insect, anything where the vertebrate
set is simply wrong — overrides the shared one by having a motif file inside its
own folder.

### What a genome folder should contain

Paralome finds these by scanning; the names do not have to match exactly:

```
Homo_sapiens.GRCh38.e113/
  Homo_sapiens.GRCh38.dna.toplevel.fa
  Homo_sapiens.GRCh38.113.gtf
  STAR/                              STAR index directory
  annovar/                           optional, for variant annotation
    humandb/
```

Name the folder `<Species>.<Assembly>.<source-initial><release>`, the name
Paralome will derive; if you do not, registration renames it. What matters more
is that the *contents* are what you think they are: a folder holding
a different annotation release from the one you confirm at registration is the
single most expensive mistake available here — results get attributed to the
wrong annotation, and nothing about the run looks wrong.

---

## 7. Adding a user later

New accounts do **not** inherit the group automatically. Nothing in
`useradd`, a web login front end, or Paralome will do it for you:

```
useradd -m -d /storage/home/carol -s /bin/bash carol
passwd carol
usermod -aG paralome carol
```

Without the group, carol has an asymmetric machine: she can read the registry,
adopt genomes and run everything, but cannot place a genome file in the
repository. The failure arrives as a bare permission error at the moment she
tries to contribute.

If the home directory already exists — created by another node on shared
storage, say — `useradd` will not copy the skeleton files into it, so she will
have no `.bashrc` or `.bash_profile`. Copy them:

```
cp -n /etc/skel/.bash* /storage/home/carol/
chown carol:carol /storage/home/carol/.bash*
```

Then have her log in once — **after** the `usermod`, so the session has the
group — and run:

```
paralome setup
```

This is required, not tidiness. It writes her config and records where the
tools are; runs use those recorded paths, so an account that has never run
`setup` has nothing to run from.

She does not need to be told where the genomes are. `site.json` tells her:

```
paralome setup genome list
```

names the repository, says whether she can write to it, and lists what is
already published.

---

## 8. What your users do

Worth knowing so you can answer questions.

```
paralome setup                      # detect tools, write a config
paralome setup genome               # what is on this machine, and where it lives
paralome setup genome sync          # adopt what they want
paralome setup doctor               # is everything in place
paralome setup doctor <genome>      # is this genome usable
paralome <ASSAY> --help             # what that analysis takes
paralome status                     # every project below here
```

`setup genome` is where people learn both things they cannot guess: the
repository path, and the exact genome names. Names are derived and long, and
they are meant to be copied from this output rather than typed.

Bare `setup genome` prints that listing, and so does `setup genome list` — the
same thing said explicitly. A name `setup doctor` does not recognise gets the
listing too, rather than a verdict: `setup doctor <typo>` says no genome is
registered under that name and then shows the ones that are, which is the only
useful reply to a name nobody registered.

**They can also just ask the command.** `paralome scRNA --help` lists what scRNA
takes, grouped by what each option is for, under one heading per stage it
runs.

`sync` alone will create a config and adopt genomes, but it does not detect
executables or the scheduler — so have people run `paralome setup` first, or
their config will carry the shipped default backend rather than the one this
machine actually has.

Adoption **copies** an entry rather than referencing it. Each person ends up
with their own record of where the files are, which is why a genome can be
removed from one account without affecting anyone else.

### Withdrawing a genome

The owner of a genome can mark it as no longer usable, with a reason everyone
else will see:

```
paralome setup genome withdraw Ovis_aries_rambouillet.ARS-UI_Ramb_v2_0.e112 \
    --superseded --reason "release 113 supersedes this"

paralome setup genome withdraw Ovis_aries_rambouillet.ARS-UI_Ramb_v2_0.e112 \
    --fatal --reason "the folder says e112, the GTF is release 113"
```

`--superseded` is the default and is nearly always the right one: it warns and
proceeds, which is correct for a genome that is simply older. hg19 is not wrong;
it is just not hg38, and years of correct results came off it.

`--fatal` refuses every future run, for everyone, including the owner. Reserve
it for a genome that produces *wrong* answers rather than old ones.

Only the account that registered a genome can withdraw it. Someone who merely
adopted a copy and no longer wants it gets what they actually meant — their copy
is removed, nothing is published, and nobody else can tell. Not needing a genome
is not evidence about the genome.

---

## 9. Common mistakes

**`chown -R user:paralome ~user`** — grants nothing, because access comes from
group membership rather than file ownership, and it labels the home with a group
it does not belong to. Use `usermod -aG`.

**Opening home directories so discovery works** — the registry exists precisely
so you do not have to. See 4.2.

**Running `setup site --apply` as anything other than root** — it refuses, and
it should: it writes a file every account on the machine then obeys.

**A per-user pip install shadowing the system one** — check with
`sudo -iu <account> which -a paralome` before believing any bug report.

**Prepending a directory of hand-built tools to PATH** — gives some accounts
different tool versions from others, and because each account records what its
own PATH found, the difference is then frozen into their configs. `paralome
setup doctor` flags a recorded path that disagrees with PATH.

**Moving or upgrading a tool and not telling anyone to re-run `paralome
setup`** — every account that recorded the old path stops with PLM-0410 on its
next run. That is the designed behaviour and it is better than the
alternative, but it arrives as a surprise if you were not expecting it. Move
tools, then have people run `paralome setup`.

**Putting the motif database outside the repository root** — the scan looks in
the genome folder and one level up, and nowhere else. Genomes registered with
it elsewhere get no motif path at all, and nothing says so until an ATAC run.

**`paralome setup --force` to refresh tools** — it does not delete anything,
but it does re-apply shipped defaults over values a user has tuned. Plain
`paralome setup` re-detects executables and the backend, which is what people
usually mean.

**Deleting a withdrawn genome's entry while people still hold copies** — that
entry is what carries the reason to everyone who adopted it. For a **fatally**
withdrawn genome Paralome refuses unless you pass `--force`, whether or not
anyone adopted it, and the refusal explains why. A superseded entry is removed
like any other (after the usual confirmation), and its warning goes with it for
everyone who adopted it. Clear the adopters
first, then the owner, then the folder — and delete the folder in the same
sitting, or the next person to register it gets a clean-looking entry with no
warning attached.

---

## 10. Getting help from the machine

```
paralome setup doctor           # the whole machine
paralome setup doctor <genome>  # one genome, in detail
paralome explain 407            # what any error code means
paralome --docs                 # refresh your copy of the docs and list it
```

Every numbered failure writes its long form to a log whose path is printed. When
a SLURM stage fails, the code you see is a relay — the real error is in the
`.err` file it names.
