from operator import itemgetter
from tree_format import format_tree
import sys as sys_arg


def print_tree(tree):
  print (format_tree(tree, format_node=itemgetter(0), get_children=itemgetter(1)))


project_tree = (
    'project/', [
      ('system/', [
        ('rdf_dir/', [
            ('mol1_mol1.xvg', []),
            ('mol1_mol2.xvg', []),
            ('mol1_mol2.xvg', []),
          ]),
        ('system_npt.edr', []),
        ('system_npt.gro', []),
        ('system.top', []),
        ],
    )
  ]
)

pc_tree = (
    'pure_components/', [
      ('molecule_temp/', [
        ('molecule1_npt.edr', []),
        ('molecule1.top', []),
        ],
    )
  ]
)

parent_tree = (
  'kbi_dir/', [
    ('project/', [
        ('system/', [
          ('rdf_dir/', [
              ('mol1_mol1.xvg', []),
              ('mol1_mol2.xvg', []),
              ('mol1_mol2.xvg', []),
            ]),
          ('system_npt.edr', []),
          ('system_npt.gro', []),
          ('system.top', []),
          ],
        )
    ]),
    ('pure_components/', [
      ('molecule1/', [
        ('molecule1_npt.edr', []),
        ('molecule1.top', []),
        ],
      )
    ]
    )
  ]
  
  )




if __name__ == "__main__":
    tree_name = sys_arg.argv[1] if len(sys_arg.argv) > 1 else 'parent'
    trees_mapped: dict[str, tuple] = {
        'project': project_tree,
        'pc': pc_tree,
        'parent': parent_tree
    }
    tree = trees_mapped.get(tree_name, parent_tree)
    print(f"Printing tree: {tree_name}")
    print_tree(tree)
