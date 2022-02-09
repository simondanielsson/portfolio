package project1.Software.Instruction;

import project1.Hardware.Operand.Operand;
import project1.Hardware.Operand.Adress.Adress;
import project1.Hardware.Operand.Word.Word;

public class Add extends BinOp {

    public Add  (Operand o1, Operand o2, Adress adress) {
        super(o1, o2, adress);
    }

    @Override
    protected void operateAndStore(Word w1, Word w2, Word storage) {
        storage.add(w1, w2); 
    }

    protected String stringOperation() {
        return "Add "; 
    }






    

}
