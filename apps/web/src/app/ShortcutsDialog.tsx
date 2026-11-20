import { Fragment } from "react";
import { Kbd, Modal } from "../components";
import { ROUTES, routeDigit } from "./navigation";

/** Each shortcut lists alternative chords; a chord is a set of keys pressed together. */
type Shortcut = { chords: string[][]; description: string };

const sc = (description: string, ...chords: string[][]): Shortcut => ({ chords, description });

const SECTIONS: Array<{ title: string; items: Shortcut[] }> = [
  {
    title: "Geral",
    items: [
      sc("Paleta de comandos", ["Ctrl", "K"]),
      sc("Recolher / expandir menu lateral", ["Ctrl", "B"]),
      sc("Mostrar estes atalhos", ["?"]),
      sc("Fechar menus, painéis e diálogos", ["Esc"]),
    ],
  },
  {
    title: "Navegação",
    items: ROUTES.map((r) => sc(r.label, ["Alt", String(routeDigit(r.id))])),
  },
  {
    title: "Biblioteca",
    items: [
      sc("Buscar na Biblioteca", ["/"]),
      sc("Mover a seleção", ["↑"], ["↓"]),
      sc("Marcar / desmarcar item", ["Espaço"], ["Ctrl", "clique"]),
      sc("Selecionar tudo (filtrado)", ["Ctrl", "A"]),
      sc("Mover para a Lixeira", ["Del"]),
      sc("Fechar inspetor / limpar seleção", ["Esc"]),
    ],
  },
  {
    title: "Menus de contexto",
    items: [
      sc("Ações do item sob o cursor", ["Botão direito"]),
      sc("Menu do item focado", ["Shift", "F10"]),
      sc("Navegar e executar no menu", ["↑"], ["↓"], ["Enter"]),
    ],
  },
];

export function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Modal open={open} onClose={onClose} title="Atalhos de teclado">
      <div className="shortcut-grid">
        {SECTIONS.map((section) => (
          <section key={section.title} className="stack stack-sm">
            <div className="section-label">{section.title}</div>
            <div>
              {section.items.map((item) => (
                <div key={item.description} className="shortcut-row">
                  <span className="secondary">{item.description}</span>
                  <span className="shortcut-keys">
                    {item.chords.map((chord, ci) => (
                      <Fragment key={ci}>
                        {ci > 0 && <span className="muted">/</span>}
                        {chord.map((key, ki) => (
                          <Fragment key={key}>
                            {ki > 0 && <span className="muted">+</span>}
                            <Kbd>{key}</Kbd>
                          </Fragment>
                        ))}
                      </Fragment>
                    ))}
                  </span>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </Modal>
  );
}
