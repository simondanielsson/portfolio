package project1.Software.Instruction;

import project1.Hardware.Instruction.Instruction;
import project1.Hardware.Memory.Memory;
import project1.Hardware.Operand.Operand;
import project1.Hardware.Operand.Adress.Adress;
import project1.Hardware.Operand.Word.Word;
import project1.Hardware.Program.ProgramCounter.ProgramCounter;

/**
 * Abstract class for uniting the responsibilities of binary operator instructions
 */
public abstract class BinOp implements Instruction {

    private Operand o1, o2; 

    private Adress adress; 

    public BinOp  (Operand o1, Operand o2, Adress adress) {
        this.o1 = o1;
        this.o2 = o2;
        this.adress = adress; 
    }

    /**
     * Perform binary operation on words w1, w2 and store the result in storage.
     * @param w1: word to be operated on
     * @param w2: word to be operated on
     * @param storage: word whose state should change into the result of the binary operation
     */
    protected abstract void operateAndStore(Word w1, Word w2, Word storage);

    public final void execute(Memory memory, ProgramCounter counter) {
        operateAndStore(
            o1.getWord(memory), 
            o2.getWord(memory),
            adress.getWord(memory)
        ); 

        // Go to next instruction in the list
        counter.step();
    } 
    
    protected abstract String stringOperation();

    @Override
    public String describe() {
        return stringOperation() + o1.toString() + " and " + o2.toString() + " into " + adress.toString();
    }
}
