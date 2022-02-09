package project1.Software.Program;

import project1.Hardware.Operand.Adress.Adress;
import project1.Hardware.Operand.Word.WordFactory.WordFactory;
import project1.Hardware.Program.Program;
import project1.Software.Instruction.*;

public class Sum extends Program {

    public Sum  (String value, WordFactory wf) {
        Adress n = new Adress(0),
                sum = new Adress(1);
        add(new Copy(wf.word(value), n));
        add(new Copy(wf.word("1"), sum));
        add(new JumpEq(6, n, wf.word("1")));
        add(new Add(sum, n, sum));
        add(new Add(n, wf.word("-1"), n));
        add(new Jump(2));
        add(new Print(sum));
        add(new Halt());
    }
}
