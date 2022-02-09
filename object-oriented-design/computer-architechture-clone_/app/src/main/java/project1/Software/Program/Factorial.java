package project1.Software.Program;

import project1.Hardware.Operand.Adress.Adress;
import project1.Hardware.Operand.Word.WordFactory.WordFactory;
import project1.Hardware.Program.Program;
import project1.Software.Instruction.*;

public class Factorial extends Program {

    public Factorial  (String value, WordFactory wf) {
        Adress n = new Adress(0),
                fac = new Adress(1);
        add(new Copy(wf.word(value), n));
        add(new Copy(wf.word("1"), fac));
        add(new JumpEq(6, n, wf.word("1")));
        add(new Mul(fac, n, fac));
        add(new Add(n, wf.word("-1"), n));
        add(new Jump(2));
        add(new Print(fac));
        add(new Halt());
    }
}
