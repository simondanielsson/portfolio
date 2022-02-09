package xl.gui;

public interface XLView {

    void updateView();

    void reportError(String errorMessage);

    void updateCurrent(String oldAddress, String newAddress);

}