package xl.gui;

import static java.awt.BorderLayout.CENTER;
import static java.awt.BorderLayout.WEST;

public class StatusPanel extends BorderPanel {

    private CurrentLabel currentLabel;

    protected StatusPanel(StatusLabel statusLabel) {
        this.currentLabel = new CurrentLabel();
        add(WEST, currentLabel);
        add(CENTER, statusLabel);
    }

    public CurrentLabel getCurrentLabel() {
        return currentLabel;
    }
}
