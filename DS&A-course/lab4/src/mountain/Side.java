package mountain;

public class Side {
	Point p1;
	Point p2;
	
	public Side(Point p1, Point p2) {
		this.p1 = p1;
		this.p2 = p2;
	}
	
	public Point getP1() {
		return p1;
	}
	
	public Point getP2() {
		return p2;
	}
	
	@Override
	public boolean equals(Object other) {
		if (other instanceof Side) {
			Side otherSide = (Side) other;
			
			if ((p1.equals(otherSide.getP1()) && p2.equals(otherSide.getP2())) ||  
				(p1.equals(otherSide.getP2()) && p2.equals(otherSide.getP1())) ) { //sidorna lika om ändpunkterna lika
				return true;
			}
		}
		return false;
	}
	
	@Override
	public int hashCode() {
		return p1.hashCode() + p2.hashCode();
	}
}
