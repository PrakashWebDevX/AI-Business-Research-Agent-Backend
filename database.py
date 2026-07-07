"""
database.py

Infrastructure layer for the AI Business Research Agent.

Responsibilities:
    - Define the SQLAlchemy ORM models for the business domain
      (Departments, Employees, Products, Customers, Orders).
    - Create the SQLite database file (database/employee.db) and all
      tables automatically if they do not already exist.
    - Seed the database with realistic sample data on first run so the
      SQL agent has meaningful data to query out of the box.
    - Expose a `get_engine()` / `get_session()` API that the rest of the
      application (sql_agent.py, agent.py) can use to talk to the database
      without knowing about connection details.

This module is idempotent: running it multiple times will not duplicate
data or recreate tables that already exist.
"""

from __future__ import annotations

import logging
import random
from datetime import date, timedelta
from pathlib import Path
from typing import List

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths & Engine configuration
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "employee.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Ensure the database directory exists before SQLite tries to create the file.
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# `check_same_thread=False` allows the connection to be safely reused across
# the different agent components that may run in separate threads (e.g. a
# web UI event loop) while SQLAlchemy still manages thread-safety via the
# session/connection pool.
engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Declarative base class shared by all ORM models."""
    pass


# --------------------------------------------------------------------------- #
# ORM Models
# --------------------------------------------------------------------------- #

class Department(Base):
    __tablename__ = "departments"

    department_id = Column(Integer, primary_key=True, autoincrement=True)
    department_name = Column(String(100), nullable=False, unique=True)
    location = Column(String(100), nullable=False)

    employees = relationship("Employee", back_populates="department", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Department id={self.department_id} name={self.department_name!r}>"


class Employee(Base):
    __tablename__ = "employees"

    employee_id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(150), nullable=False, unique=True)
    phone = Column(String(20), nullable=False)
    job_title = Column(String(100), nullable=False)
    hire_date = Column(Date, nullable=False)
    salary = Column(Numeric(10, 2), nullable=False)
    department_id = Column(Integer, ForeignKey("departments.department_id"), nullable=False)

    department = relationship("Department", back_populates="employees")
    orders = relationship("Order", back_populates="employee")

    def __repr__(self) -> str:
        return f"<Employee id={self.employee_id} name={self.first_name} {self.last_name!r}>"


class Product(Base):
    __tablename__ = "products"

    product_id = Column(Integer, primary_key=True, autoincrement=True)
    product_name = Column(String(150), nullable=False, unique=True)
    category = Column(String(100), nullable=False)
    unit_price = Column(Numeric(10, 2), nullable=False)
    stock_quantity = Column(Integer, nullable=False)

    orders = relationship("Order", back_populates="product")

    def __repr__(self) -> str:
        return f"<Product id={self.product_id} name={self.product_name!r}>"


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(150), nullable=False, unique=True)
    phone = Column(String(20), nullable=False)
    city = Column(String(100), nullable=False)
    country = Column(String(100), nullable=False)

    orders = relationship("Order", back_populates="customer", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Customer id={self.customer_id} name={self.first_name} {self.last_name!r}>"


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), nullable=False)
    employee_id = Column(Integer, ForeignKey("employees.employee_id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.product_id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    total_amount = Column(Numeric(10, 2), nullable=False)
    order_date = Column(Date, nullable=False)
    status = Column(String(20), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    customer = relationship("Customer", back_populates="orders")
    employee = relationship("Employee", back_populates="orders")
    product = relationship("Product", back_populates="orders")

    def __repr__(self) -> str:
        return f"<Order id={self.order_id} customer_id={self.customer_id} total={self.total_amount}>"


# --------------------------------------------------------------------------- #
# Sample data pools (used to generate realistic, non-random-looking records)
# --------------------------------------------------------------------------- #

_FIRST_NAMES: List[str] = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Daniel", "Nancy", "Matthew", "Lisa",
    "Anthony", "Betty", "Mark", "Margaret", "Paul", "Sandra", "Steven", "Ashley",
    "Andrew", "Kimberly", "Kenneth", "Emily", "George", "Donna", "Joshua", "Michelle",
    "Kevin", "Priya", "Brian", "Aisha", "Ravi", "Chen", "Wei", "Fatima", "Carlos", "Sofia",
]

_LAST_NAMES: List[str] = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
]

_DEPARTMENTS: List[dict] = [
    {"department_name": "Sales", "location": "New York, USA"},
    {"department_name": "Marketing", "location": "Chicago, USA"},
    {"department_name": "Engineering", "location": "San Francisco, USA"},
    {"department_name": "Finance", "location": "Boston, USA"},
    {"department_name": "Human Resources", "location": "Austin, USA"},
    {"department_name": "Customer Support", "location": "Denver, USA"},
    {"department_name": "Operations", "location": "Seattle, USA"},
]

_JOB_TITLES: List[str] = [
    "Sales Representative", "Account Manager", "Marketing Specialist",
    "Software Engineer", "Senior Software Engineer", "Financial Analyst",
    "HR Coordinator", "Customer Support Agent", "Operations Manager",
    "Business Analyst", "Product Manager", "Data Analyst",
]

_PRODUCTS: List[dict] = [
    {"product_name": "Wireless Mouse", "category": "Electronics", "unit_price": 19.99},
    {"product_name": "Mechanical Keyboard", "category": "Electronics", "unit_price": 59.99},
    {"product_name": "27-inch Monitor", "category": "Electronics", "unit_price": 249.99},
    {"product_name": "USB-C Docking Station", "category": "Electronics", "unit_price": 89.99},
    {"product_name": "Noise Cancelling Headphones", "category": "Electronics", "unit_price": 129.99},
    {"product_name": "Ergonomic Office Chair", "category": "Furniture", "unit_price": 219.99},
    {"product_name": "Standing Desk", "category": "Furniture", "unit_price": 349.99},
    {"product_name": "Desk Lamp", "category": "Furniture", "unit_price": 24.99},
    {"product_name": "Business Laptop Bag", "category": "Accessories", "unit_price": 44.99},
    {"product_name": "Portable SSD 1TB", "category": "Electronics", "unit_price": 99.99},
    {"product_name": "Whiteboard 4x6", "category": "Office Supplies", "unit_price": 74.99},
    {"product_name": "Printer - LaserJet", "category": "Electronics", "unit_price": 179.99},
]

_CITIES_COUNTRIES: List[dict] = [
    {"city": "New York", "country": "USA"},
    {"city": "Los Angeles", "country": "USA"},
    {"city": "London", "country": "UK"},
    {"city": "Manchester", "country": "UK"},
    {"city": "Toronto", "country": "Canada"},
    {"city": "Vancouver", "country": "Canada"},
    {"city": "Sydney", "country": "Australia"},
    {"city": "Melbourne", "country": "Australia"},
    {"city": "Berlin", "country": "Germany"},
    {"city": "Mumbai", "country": "India"},
    {"city": "Bengaluru", "country": "India"},
    {"city": "Singapore", "country": "Singapore"},
    {"city": "Dubai", "country": "UAE"},
    {"city": "Paris", "country": "France"},
    {"city": "Tokyo", "country": "Japan"},
]

_ORDER_STATUSES: List[str] = ["Pending", "Completed", "Shipped", "Cancelled"]

# Seed counts (minimums as required)
NUM_EMPLOYEES = 30
NUM_CUSTOMERS = 20
NUM_ORDERS = 50


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _random_date(start: date, end: date) -> date:
    """Return a random date between start and end (inclusive)."""
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def _unique_email(first_name: str, last_name: str, index: int, domain: str) -> str:
    """Build a realistic, guaranteed-unique email address."""
    return f"{first_name.lower()}.{last_name.lower()}{index}@{domain}"


# --------------------------------------------------------------------------- #
# Seeding logic
# --------------------------------------------------------------------------- #

def _seed_departments(session: Session) -> List[Department]:
    departments = [Department(**data) for data in _DEPARTMENTS]
    session.add_all(departments)
    session.flush()  # populate department_id without committing yet
    logger.info("Seeded %d departments.", len(departments))
    return departments


def _seed_employees(session: Session, departments: List[Department]) -> List[Employee]:
    employees: List[Employee] = []
    hire_start, hire_end = date(2015, 1, 1), date(2026, 6, 1)

    for i in range(1, NUM_EMPLOYEES + 1):
        first_name = random.choice(_FIRST_NAMES)
        last_name = random.choice(_LAST_NAMES)
        employees.append(
            Employee(
                first_name=first_name,
                last_name=last_name,
                email=_unique_email(first_name, last_name, i, "company.com"),
                phone=f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                job_title=random.choice(_JOB_TITLES),
                hire_date=_random_date(hire_start, hire_end),
                salary=round(random.uniform(45_000, 145_000), 2),
                department_id=random.choice(departments).department_id,
            )
        )
    session.add_all(employees)
    session.flush()
    logger.info("Seeded %d employees.", len(employees))
    return employees


def _seed_products(session: Session) -> List[Product]:
    products: List[Product] = []
    for data in _PRODUCTS:
        products.append(
            Product(
                product_name=data["product_name"],
                category=data["category"],
                unit_price=data["unit_price"],
                stock_quantity=random.randint(10, 500),
            )
        )
    session.add_all(products)
    session.flush()
    logger.info("Seeded %d products.", len(products))
    return products


def _seed_customers(session: Session) -> List[Customer]:
    customers: List[Customer] = []
    for i in range(1, NUM_CUSTOMERS + 1):
        first_name = random.choice(_FIRST_NAMES)
        last_name = random.choice(_LAST_NAMES)
        location = random.choice(_CITIES_COUNTRIES)
        customers.append(
            Customer(
                first_name=first_name,
                last_name=last_name,
                email=_unique_email(first_name, last_name, i, "example.com"),
                phone=f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                city=location["city"],
                country=location["country"],
            )
        )
    session.add_all(customers)
    session.flush()
    logger.info("Seeded %d customers.", len(customers))
    return customers


def _seed_orders(
    session: Session,
    customers: List[Customer],
    employees: List[Employee],
    products: List[Product],
) -> List[Order]:
    orders: List[Order] = []
    order_start, order_end = date(2024, 1, 1), date(2026, 6, 30)

    for _ in range(NUM_ORDERS):
        product = random.choice(products)
        quantity = random.randint(1, 10)
        total_amount = round(float(product.unit_price) * quantity, 2)

        orders.append(
            Order(
                customer_id=random.choice(customers).customer_id,
                employee_id=random.choice(employees).employee_id,
                product_id=product.product_id,
                quantity=quantity,
                total_amount=total_amount,
                order_date=_random_date(order_start, order_end),
                status=random.choice(_ORDER_STATUSES),
            )
        )
    session.add_all(orders)
    session.flush()
    logger.info("Seeded %d orders.", len(orders))
    return orders


def _database_is_empty(session: Session) -> bool:
    """Check whether the core tables already contain data."""
    return session.query(Employee).first() is None


def seed_database(session: Session) -> None:
    """
    Populate the database with realistic sample data.

    This function is idempotent: it only seeds data if the Employees
    table is currently empty, so re-running the app will not create
    duplicate records.
    """
    if not _database_is_empty(session):
        logger.info("Database already contains data. Skipping seeding.")
        return

    logger.info("Seeding database with sample data...")
    departments = _seed_departments(session)
    employees = _seed_employees(session, departments)
    products = _seed_products(session)
    customers = _seed_customers(session)
    _seed_orders(session, customers, employees, products)

    session.commit()
    logger.info("Database seeding complete.")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_engine():
    """Return the shared SQLAlchemy engine bound to employee.db."""
    return engine


def get_session() -> Session:
    """Return a new SQLAlchemy session for interacting with the database."""
    return SessionLocal()


def init_db() -> None:
    """
    Initialize the database: create employee.db and all tables if they
    do not exist yet, then seed sample data if the database is empty.

    This is the single entry point the rest of the application
    (sql_agent.py, agent.py, app.py) should call on startup.
    """
    logger.info("Initializing database at %s", DATABASE_PATH)
    Base.metadata.create_all(bind=engine)

    with get_session() as session:
        seed_database(session)

    logger.info("Database ready.")


if __name__ == "__main__":
    # Allows running `python database.py` directly to set up the database
    # without starting the full application.
    init_db()
