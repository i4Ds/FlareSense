# Pro5D - Klassifikation von Sonneneruptionen mittels Machine Learning und e-Callisto Netzwerk
Dieses Projekt wird im Rahmen der Studiengangs B.Sc. Data Science der Fachhochschule Nordwestschweiz bearbeitet.

## Projektarbeiter
Dieses Projekt wird von folgenden Studenten bearbeitet:
| Name                 | GitHub Handle                                                |
| :------------------- | :----------------------------------------------------------- |
| Patrick Schürmann    | [@patschue](https://github.com/patschue)                     |
| Gabriel Torres Gamez | [@gabrieltorresgamez](https://github.com/gabrieltorresgamez) |

## Dokumentation
Hier geht es zur Dokumentation:

[Link zur Wiki](https://github.com/i4Ds/FlareSense/wiki)

## Taskboard
Hier geht es zum Projekt Taskboard mit allen Tasks und Issues:

[Link zum Taskboard](https://github.com/orgs/i4Ds/projects/11)

## Diskussionen
Um Fragen, Ideen oder Wünsche zu stellen, kann man das bei den "Discussions" machen:

[Link zu den Discussions](https://github.com/i4Ds/FlareSense/discussions)

## Zotero
Hier geht es zu unserer Zotero Gruppe:

[Link zu Zotero](https://www.zotero.org/groups/5202251/pro5d_23hs_i4ds22/library)

## DagsHub
Hier geht es zu unserer DagsHub Organisation: 

[Link zur Organisation](https://dagshub.com/org/FlareSense/home)

## Repository Setup Instructions
1. Clone the repo.
2. Run `make reqs` to install required python packages.

### If you want to use DagsHub:
3. Setup the DVC credentials using DagsHub.
![Setup DVC Credentials](https://i.imgur.com/BgCl22U.png)
4. Run `make pull` to pull the data from DagsHub.
5. You're ready to start developing!

### Datamanagement with DagsHub
The data in this repo is managed via DVC. Here are some useful commands:
- `make pull` - Pulls the data from DagsHub.
- `make relink` - After changes in the DVC Folders (data) this command relinks the files in the repo.
- `make push` - Pushes the data to DagsHub.

Before committing changes to the data folder, make sure to run `make relink` to update the links to the data.
Afterwards, add, commit and push the changes to the repo (using git).
Finally, run `make push` to push the data to DagsHub.

## Project Organization

    ├── .dvc               <- DVC Settings, don't touch.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering)
    │                         and a short `-` delimited description, e.g.
    │                         `01-initial-data-exploration`.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── slurm              <- Slurm scripts for running the code on the i4Ds cluster.
    ├── src                <- Source code for use in this project.
    ├── .dvcignore         <- Files and directories to ignore by DVC.
    ├── .gitignore         <- Files and directories to ignore by Git.
    ├── data.dvc           <- DVC data/ folder registry.
    ├── LICENSE            <- GNU General Public License v3.0.
    ├── Makefile           <- Makefile with commands.
    ├── params.yml        <- The parameters for the data pipeline.
    ├── README.md          <- The top-level README for developers using this project.
    └── requirements.txt   <- The requirements file for reproducing the analysis environment.
